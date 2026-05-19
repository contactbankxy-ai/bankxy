"""
=============================================================================
BANKXY — Flask Server (Production-grade)
=============================================================================
Pipeline per request:
  1. Receive PDF
  2. Detect bank (auto or manual)
  3. Run parser chain with fallbacks
  4. Run ValidationEngine (7-layer check + confidence score)
  5. Apply safe-export policy
  6. Return JSON with validation report, preview, and Base64 Excel
=============================================================================
"""

import io
import os
import base64
import json
import logging
import shutil
import tempfile
import time
import uuid
import gc
from datetime import datetime

from flask import Flask, jsonify, render_template, request, send_from_directory

from bank_parsers import detect_bank, run_parser_chain
from validation_engine import ValidationEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("bankxy.server")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
COUNT_FILE  = os.path.join(BASE_DIR, "pdf_count.json")
IP_FILE     = os.path.join(BASE_DIR, "ip_pdf_count.json")
FAILED_DIR  = os.path.join(BASE_DIR, "failed_pdfs")
os.makedirs(FAILED_DIR, exist_ok=True)

EXCEL_DIR = os.path.join(BASE_DIR, "temp_excel")
os.makedirs(EXCEL_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB upload limit to prevent OOM

import database
from firebase_admin import auth
from functools import wraps

# ---------------------------------------------------------------------------
# Admin Security Decorator
# ---------------------------------------------------------------------------
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Unauthorized"}), 401
        
        token = auth_header.split(' ')[1]
        try:
            decoded_token = auth.verify_id_token(token)
            email = decoded_token.get('email', '')
            
            db = database.get_db()
            if not db:
                return jsonify({"error": "Database not configured"}), 500
                
            user_doc = db.collection("users").document(email).get()
            if not user_doc.exists or user_doc.to_dict().get("role") != "admin":
                return jsonify({"error": "Forbidden"}), 403
                
        except Exception as e:
            return jsonify({"error": f"Invalid token: {str(e)}"}), 403
            
        return f(*args, **kwargs)
    return decorated_function

# Validation engine (singleton)
# ---------------------------------------------------------------------------
validator = ValidationEngine()

# ---------------------------------------------------------------------------
# Failed-PDF logger
# ---------------------------------------------------------------------------
def _log_failed_pdf(source_path: str, bank: str, reason: str):
    """Copy PDF to failed_pdfs/ with a timestamped name and a .txt log."""
    try:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{bank}_{ts}"
        dst  = os.path.join(FAILED_DIR, f"{name}.pdf")
        shutil.copy2(source_path, dst)
        with open(os.path.join(FAILED_DIR, f"{name}.txt"), "w") as f:
            f.write(f"Bank: {bank}\nReason: {reason}\nTimestamp: {ts}\n")
        logger.info("Failed PDF saved → %s", dst)
    except Exception as e:
        logger.warning("Could not save failed PDF: %s", e)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/test-pyodide")
def test_pyodide():
    return render_template("pyodide_test.html")

@app.route("/src/<path:filename>")
def serve_src(filename):
    # Security: only allow serving specific python files for the PoC
    allowed_files = ["bank_parsers.py", "validation_engine.py"]
    if filename in allowed_files:
        return send_from_directory(BASE_DIR, filename, mimetype="text/plain")
    return "Not allowed", 403

@app.route("/api/check_admin", methods=["GET"])
def check_admin():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"isAdmin": False})
    
    token = auth_header.split(' ')[1]
    try:
        decoded_token = auth.verify_id_token(token)
        email = decoded_token.get('email', '')
        
        db = database.get_db()
        if not db:
            return jsonify({"isAdmin": False})
            
        # Log login to ensure user document exists for easy editing in Firestore Console
        database.log_user_login(email)
            
        user_doc = db.collection("users").document(email).get()
        if user_doc.exists and user_doc.to_dict().get("role") == "admin":
            return jsonify({"isAdmin": True})
            
    except Exception as e:
        logger.warning(f"Check admin error: {e}")
        pass
        
    return jsonify({"isAdmin": False})

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/api/admin/dashboard_data")
@admin_required
def admin_dashboard_data():
    db = database.get_db()
    if not db:
        return jsonify({
            "total_users": 0, "total_conversions": 0, "total_errors": 0,
            "recent_activity": [], "recent_errors": [],
            "chart_data": { "dates": [], "counts": [], "banks": { "labels": [], "data": [] } },
            "warning": "Firebase not configured. Missing firebase-admin-key.json"
        })
        
    try:
        users_count = db.collection("users").count().get()[0][0].value
        conv_count = db.collection("conversions").count().get()[0][0].value
        err_count = db.collection("errors").count().get()[0][0].value
    except:
        users_count, conv_count, err_count = 0, 0, 0
        
    recent_activity = []
    for c in db.collection("conversions").order_by("timestamp", direction="DESCENDING").limit(10).stream():
        d = c.to_dict()
        ts = d.get('timestamp')
        d['timestamp'] = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else ""
        recent_activity.append(d)
        
    recent_errors = []
    for e in db.collection("errors").order_by("timestamp", direction="DESCENDING").limit(10).stream():
        d = e.to_dict()
        ts = d.get('timestamp')
        d['timestamp'] = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else ""
        recent_errors.append(d)
        
    dates_map = {}
    banks_map = {}
    for doc in db.collection("conversions").order_by("timestamp", direction="DESCENDING").limit(100).stream():
        d = doc.to_dict()
        b = d.get("bank", "Unknown")
        banks_map[b] = banks_map.get(b, 0) + 1
        ts = d.get("timestamp")
        if ts:
            dt_str = ts.strftime("%Y-%m-%d")
            dates_map[dt_str] = dates_map.get(dt_str, 0) + 1

    sorted_dates = sorted(dates_map.keys())[-7:]
    counts = [dates_map[d] for d in sorted_dates]
    
    return jsonify({
        "total_users": users_count,
        "total_conversions": conv_count,
        "total_errors": err_count,
        "recent_activity": recent_activity,
        "recent_errors": recent_errors,
        "chart_data": {
            "dates": sorted_dates,
            "counts": counts,
            "banks": { "labels": list(banks_map.keys()), "data": list(banks_map.values()) }
        }
    })

@app.route("/api/convert", methods=["POST"])
def api_convert():
    # ── 1. Basic file validation ──────────────────────────────────────────
    if "pdf_file" not in request.files:
        return jsonify({"status": "error", "message": "No PDF file provided."}), 400

    file = request.files["pdf_file"]
    if not file.filename:
        return jsonify({"status": "error", "message": "Empty filename."}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({
            "status": "error",
            "message": "Invalid file type. Only PDF files are accepted.",
        }), 400

    password         = request.form.get("password", "").strip() or None
    requested_bank   = request.form.get("bank", "auto").lower().strip()

    # ── 2. Save to temp file ──────────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        file.save(tmp.name)
        tmp.close()

        # ── 3. Bank detection ─────────────────────────────────────────────
        if requested_bank in ("auto", ""):
            detected_bank = detect_bank(tmp.name, password)
            logger.info("Auto-detected bank: %s", detected_bank)
        else:
            detected_bank = requested_bank
            logger.info("Manually selected bank: %s", detected_bank)

        # ── 4. Parser chain ───────────────────────────────────────────────
        try:
            df, page_count, parser_used = run_parser_chain(detected_bank, tmp.name, password)
        except Exception as e:
            _log_failed_pdf(tmp.name, detected_bank, str(e))
            database.log_error(detected_bank, str(e), "guest")
            database.log_conversion(detected_bank, "failed", os.path.getsize(tmp.name), "guest", False, 0)
            return jsonify({
                "status": "error",
                "message": (
                    "Could not extract any transactions from this PDF. "
                    "If the file is encrypted, please provide the correct password. "
                    f"Technical detail: {e}"
                ),
            }), 400

        if df is None or df.empty:
            _log_failed_pdf(tmp.name, detected_bank, "Empty DataFrame after parsing")
            database.log_error(detected_bank, "Empty DataFrame after parsing", "guest")
            database.log_conversion(detected_bank, "failed", os.path.getsize(tmp.name), "guest", False, 0)
            return jsonify({
                "status": "error",
                "message": "No transactions found in this document.",
            }), 400

        # ── 5. Validation Engine ──────────────────────────────────────────
        df, report = validator.run(df, tmp.name, page_count, password)

        report_dict = report.to_dict()
        report_dict["parser_used"] = parser_used
        report_dict["detected_bank"] = detected_bank.upper()

        # ── 6. Safe-export policy ─────────────────────────────────────────
        if not report.export_allowed:
            _log_failed_pdf(tmp.name, detected_bank, report.block_reason)
            database.log_error(detected_bank, report.block_reason, "guest")
            database.log_conversion(detected_bank, "blocked", os.path.getsize(tmp.name), "guest", report.is_scanned_pdf, 0)
            return jsonify({
                "status": "blocked",
                "message": report.block_reason,
                "validation_report": report_dict,
            }), 422

        # ── 7. Build Excel to disk instead of memory ──────────────────────
        file_id = str(uuid.uuid4())
        excel_path = os.path.join(EXCEL_DIR, f"{file_id}.xlsx")
        df.to_excel(excel_path, index=False)

        # Preview (top 5 rows, NaN → empty string)
        preview_df = df.head(5).fillna("")
        preview = {
            "columns": preview_df.columns.tolist(),
            "rows":    preview_df.values.tolist(),
        }

        # ── 8. Update database ────────────────────────────────────────────
        database.log_conversion(detected_bank.upper(), "success", os.path.getsize(tmp.name), "guest", report.is_scanned_pdf, 0)

        # Use original filename but change extension to .xlsx
        original_filename = file.filename
        base_name = os.path.splitext(original_filename)[0]
        excel_filename = f"{base_name}.xlsx"

        return jsonify({
            "status":            "success",
            "message":           "Statement processed and validated successfully.",
            "preview":           preview,
            "filename":          excel_filename,
            "file_id":           file_id,
            "validation_report": report_dict,
        })

    except Exception as e:
        logger.exception("Unexpected error in /api/convert")
        return jsonify({
            "status":  "error",
            "message": "An unexpected server error occurred. The team has been notified.",
        }), 500

    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass
        
        # Periodic cleanup of old temp excel files (older than 1 hour)
        try:
            now = time.time()
            for f in os.listdir(EXCEL_DIR):
                fpath = os.path.join(EXCEL_DIR, f)
                if os.path.isfile(fpath) and os.stat(fpath).st_mtime < now - 3600:
                    os.remove(fpath)
        except Exception:
            pass
            
        # Explicit garbage collection to free memory
        gc.collect()

@app.route("/api/download/<file_id>")
def download_excel(file_id):
    filename = request.args.get("name", "Statement.xlsx")
    excel_path = os.path.join(EXCEL_DIR, f"{file_id}.xlsx")
    if not os.path.exists(excel_path):
        return "File not found or expired.", 404
    return send_from_directory(EXCEL_DIR, f"{file_id}.xlsx", as_attachment=True, download_name=filename)

# ── Admin / ops routes ─────────────────────────────────────────────────────
@app.route("/ip-stats")
def ip_stats():
    rows = "".join(f"<li>{ip} → {cnt}</li>" for ip, cnt in ip_counts.items())
    return f"<h2>IP Stats</h2><ul>{rows}</ul>"

@app.route("/failed-pdfs")
def list_failed():
    files = os.listdir(FAILED_DIR)
    items = "".join(f"<li>{f}</li>" for f in sorted(files))
    return f"<h2>Failed PDFs ({len(files)})</h2><ul>{items}</ul>"

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
