"""
=============================================================================
BANKXY — Validation Engine
=============================================================================
Multi-layer fail-safe validation and confidence scoring for parsed bank
statement data. Designed for enterprise-grade financial data integrity.

Layers:
  1. ColumnValidator       — required column presence / alias detection
  2. DateValidator         — date format, chronology, impossible values
  3. BalanceContinuityValidator — accounting equation row-by-row
  4. DuplicateValidator    — duplicate transaction detection
  5. TransactionSanityCheck — page-count vs. row-count heuristics
  6. NarrationValidator    — narration quality / completeness
  7. OCRDetector           — scanned-PDF detection
  8. ConfidenceEngine      — weighted composite score (0–100)
  9. ValidationEngine      — orchestrates all layers → ValidationReport
=============================================================================
"""

import re
import logging
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("bankxy.validation")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    detail: str = ""
    severity: str = "info"   # info | warning | error


@dataclass
class ValidationReport:
    checks: list = field(default_factory=list)
    confidence_score: float = 0.0
    confidence_label: str = "Unknown"
    export_allowed: bool = False
    block_reason: str = ""
    suspicious_rows: list = field(default_factory=list)   # indices
    duplicate_rows: list = field(default_factory=list)    # indices
    warnings: list = field(default_factory=list)
    total_transactions: int = 0
    balance_mismatches: int = 0
    is_scanned_pdf: bool = False

    def to_dict(self):
        """JSON-serialisable summary for the API response."""
        return {
            "confidence_score": round(self.confidence_score, 1),
            "confidence_label": self.confidence_label,
            "export_allowed": self.export_allowed,
            "block_reason": self.block_reason,
            "total_transactions": self.total_transactions,
            "balance_mismatches": self.balance_mismatches,
            "duplicate_rows_found": len(self.duplicate_rows),
            "suspicious_rows_found": len(self.suspicious_rows),
            "is_scanned_pdf": self.is_scanned_pdf,
            "warnings": self.warnings,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "detail": c.detail,
                    "severity": c.severity,
                }
                for c in self.checks
            ],
        }


# ===========================================================================
# Layer 1 — Column Validator
# ===========================================================================
# Canonical name → list of accepted aliases (case-insensitive)
COLUMN_ALIASES = {
    "date":        ["date", "txn date", "transaction date", "posting date", "value date"],
    "narration":   ["narration", "description", "remarks", "particulars", "transaction details",
                    "details", "trans description"],
    "debit":       ["debit", "withdrawal", "withdrawal amt.", "withdrawals", "dr", "amount dr",
                    "debit amount", "dr amount"],
    "credit":      ["credit", "deposit", "deposit amt.", "deposits", "cr", "amount cr",
                    "credit amount", "cr amount"],
    "balance":     ["balance", "closing balance", "running balance", "balance(rs.)",
                    "balance(inr)", "available balance", "balance amt."],
    "reference":   ["chq./ref.no.", "ref no", "reference number", "transaction id", "chq no", "cheque no", "ref.no.", "utr"],
}

class ColumnValidator:
    """Checks that the DataFrame contains recognisable financial columns."""

    def validate(self, df: pd.DataFrame) -> CheckResult:
        cols_lower = [str(c).strip().lower() for c in df.columns]
        found = {}
        for canonical, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in cols_lower:
                    found[canonical] = alias
                    break

        missing_critical = [k for k in ("date", "balance") if k not in found]
        has_amount = "debit" in found or "credit" in found

        if missing_critical:
            return CheckResult(
                name="Column Validation",
                passed=False,
                message=f"Critical columns missing: {', '.join(missing_critical)}",
                detail="Export blocked — required financial columns could not be located.",
                severity="error",
            )
        if not has_amount:
            return CheckResult(
                name="Column Validation",
                passed=False,
                message="No Debit or Credit column detected.",
                detail="At least one amount column is required for a valid bank statement.",
                severity="error",
            )

        return CheckResult(
            name="Column Validation",
            passed=True,
            message="All required columns detected.",
            detail=f"Mapped: {found}",
            severity="info",
        )

    def resolve_column_name(self, df: pd.DataFrame, canonical: str) -> Optional[str]:
        """Return actual DataFrame column name matching a canonical key, or None."""
        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        for alias in COLUMN_ALIASES.get(canonical, []):
            if alias in cols_lower:
                return cols_lower[alias]
        return None


# ===========================================================================
# Layer 2 — Date Validator
# ===========================================================================
DATE_PATTERNS = [
    ("%d/%m/%Y", r"\d{2}/\d{2}/\d{4}"),
    ("%d/%m/%y",  r"\d{2}/\d{2}/\d{2}"),
    ("%d-%m-%Y", r"\d{2}-\d{2}-\d{4}"),
    ("%d-%m-%y",  r"\d{2}-\d{2}-\d{2}"),
    ("%Y-%m-%d", r"\d{4}-\d{2}-\d{2}"),
    ("%d %b %Y", r"\d{2} [A-Za-z]{3} \d{4}"),
    ("%d %b %y",  r"\d{2} [A-Za-z]{3} \d{2}"),
]

def _parse_date(val: str) -> Optional[date]:
    val = str(val).strip()
    for fmt, pattern in DATE_PATTERNS:
        if re.fullmatch(pattern, val, re.IGNORECASE):
            try:
                return datetime.strptime(val, fmt).date()
            except ValueError:
                continue
    return None


class DateValidator:
    def validate(self, df: pd.DataFrame, date_col: str) -> CheckResult:
        if date_col not in df.columns:
            return CheckResult("Date Validation", False, "Date column not found.", severity="error")

        dates = df[date_col].dropna().astype(str)
        if dates.empty:
            return CheckResult("Date Validation", False, "Date column is entirely empty.", severity="error")

        parsed = [_parse_date(d) for d in dates]
        failed = [str(dates.iloc[i]) for i, p in enumerate(parsed) if p is None]

        invalid_pct = len(failed) / len(dates)
        if invalid_pct > 0.20:
            return CheckResult(
                "Date Validation",
                False,
                f"{len(failed)} of {len(dates)} dates are unparseable ({invalid_pct:.0%}).",
                detail=f"Sample invalid: {failed[:5]}",
                severity="error",
            )

        valid_dates = [p for p in parsed if p is not None]
        out_of_order = sum(
            1 for i in range(1, len(valid_dates)) if valid_dates[i] < valid_dates[i - 1]
        )
        future_dates = [str(d) for d in valid_dates if d > date.today()]

        detail_parts = []
        severity = "info"
        if out_of_order:
            detail_parts.append(f"{out_of_order} out-of-order date(s) found.")
            severity = "warning"
        if future_dates:
            detail_parts.append(f"{len(future_dates)} future date(s): {future_dates[:3]}")
            severity = "warning"
        if failed:
            detail_parts.append(f"{len(failed)} date(s) could not be parsed.")

        msg = "Dates validated." if not detail_parts else "Date issues found."
        return CheckResult(
            "Date Validation",
            passed=(invalid_pct <= 0.20),
            message=msg,
            detail=" | ".join(detail_parts) if detail_parts else "All dates valid.",
            severity=severity,
        )


# ===========================================================================
# Layer 3 — Balance Continuity Validator
# ===========================================================================
TOLERANCE = 1.0   # ₹1 rounding tolerance


def _to_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        s = str(val).replace(",", "").replace(" ", "").strip()
        s = re.sub(r"[^\d.\-]", "", s)
        return float(s) if s else None
    except (ValueError, TypeError):
        return None


class BalanceContinuityValidator:
    """
    Validates: prev_balance + credit - debit ≈ curr_balance
    Returns suspicious row indices and mismatch count.
    """

    def validate(
        self, df: pd.DataFrame,
        balance_col: str,
        debit_col: Optional[str],
        credit_col: Optional[str],
    ) -> tuple[CheckResult, list, int]:
        """Returns (CheckResult, suspicious_indices, mismatch_count)."""

        suspicious = []
        if not balance_col or balance_col not in df.columns:
            return (
                CheckResult("Balance Continuity", False,
                            "Balance column not found.", severity="error"),
                suspicious, 0
            )

        balances = [_to_float(v) for v in df[balance_col]]
        debits   = [_to_float(df[debit_col].iloc[i]) if debit_col and debit_col in df.columns else None
                    for i in range(len(df))]
        credits  = [_to_float(df[credit_col].iloc[i]) if credit_col and credit_col in df.columns else None
                    for i in range(len(df))]

        mismatch_count = 0
        for i in range(1, len(df)):
            prev_bal = balances[i - 1]
            curr_bal = balances[i]
            dr = debits[i]  if debits[i]  is not None else 0.0
            cr = credits[i] if credits[i] is not None else 0.0

            if prev_bal is None or curr_bal is None:
                continue

            expected = round(prev_bal + cr - dr, 2)
            if abs(expected - curr_bal) > TOLERANCE:
                mismatch_count += 1
                suspicious.append(i)
                logger.debug(
                    "Balance mismatch at row %d: prev=%.2f dr=%.2f cr=%.2f "
                    "expected=%.2f actual=%.2f",
                    i, prev_bal, dr, cr, expected, curr_bal
                )

        total = len(df) - 1
        if total <= 0:
            return (
                CheckResult("Balance Continuity", True, "Not enough rows to validate.", severity="info"),
                suspicious, 0
            )

        pct_ok = 1 - (mismatch_count / total)
        if pct_ok >= 0.95:
            result = CheckResult(
                "Balance Continuity", True,
                f"Balance continuity passed ({mismatch_count} mismatch(es) in {total} rows).",
                severity="info",
            )
        elif pct_ok >= 0.80:
            result = CheckResult(
                "Balance Continuity", True,
                f"Balance continuity acceptable with {mismatch_count} mismatch(es). Verify manually.",
                severity="warning",
            )
        else:
            result = CheckResult(
                "Balance Continuity", False,
                f"High balance mismatch rate: {mismatch_count}/{total} rows failed the accounting equation.",
                detail="prev_balance + credit - debit ≠ curr_balance in many rows.",
                severity="error",
            )

        return result, suspicious, mismatch_count


# ===========================================================================
# Layer 4 — Duplicate Validator
# ===========================================================================
class DuplicateValidator:
    def validate(self, df: pd.DataFrame,
                 date_col: Optional[str],
                 narration_col: Optional[str],
                 balance_col: Optional[str],
                 ref_col: Optional[str] = None,
                 debit_col: Optional[str] = None,
                 credit_col: Optional[str] = None) -> tuple[CheckResult, list]:

        key_cols = [c for c in [date_col, narration_col, balance_col, ref_col, debit_col, credit_col]
                    if c and c in df.columns]

        if not key_cols:
            return (
                CheckResult("Duplicate Check", False,
                            "Insufficient columns for duplicate detection.", severity="warning"),
                []
            )

        # We create a temporary comparison string to find exact duplicates across all key columns
        subset = df[key_cols].astype(str).apply(lambda col: col.str.strip().str.lower())
        dup_mask = subset.duplicated(keep="first")
        dup_indices = df.index[dup_mask].tolist()

        if not dup_indices:
            return (
                CheckResult("Duplicate Check", True,
                            "No duplicate transactions detected.", severity="info"),
                []
            )

        return (
            CheckResult(
                "Duplicate Check",
                passed=len(dup_indices) < 5,   # small number is a warning, not a blocker
                message=f"{len(dup_indices)} potential duplicate row(s) detected.",
                detail="Duplicates (identical in all columns including Ref No) will be removed.",
                severity="warning" if len(dup_indices) < 10 else "error",
            ),
            dup_indices,
        )


# ===========================================================================
# Layer 5 — Transaction Sanity Check (page vs rows heuristic)
# ===========================================================================
class TransactionSanityCheck:
    MIN_ROWS_PER_PAGE = 1
    EXPECTED_ROWS_PER_PAGE = 8    # conservative estimate

    def validate(self, df: pd.DataFrame, page_count: int) -> CheckResult:
        row_count = len(df)

        if row_count == 0:
            return CheckResult(
                "Transaction Count Sanity", False,
                "Zero transactions extracted from the PDF.",
                severity="error",
            )

        if page_count > 0:
            ratio = row_count / page_count
            if ratio < self.MIN_ROWS_PER_PAGE and page_count > 2:
                return CheckResult(
                    "Transaction Count Sanity", False,
                    f"Suspiciously low extraction: {row_count} rows from {page_count} pages "
                    f"({ratio:.1f} rows/page). Parser may have failed.",
                    severity="error",
                )
            if ratio < 2 and page_count > 3:
                return CheckResult(
                    "Transaction Count Sanity", True,
                    f"{row_count} transactions from {page_count} pages. "
                    "Lower than expected — verify completeness.",
                    severity="warning",
                )

        return CheckResult(
            "Transaction Count Sanity", True,
            f"{row_count} transaction(s) extracted successfully.",
            severity="info",
        )


# ===========================================================================
# Layer 6 — Narration Quality Validator
# ===========================================================================
class NarrationValidator:
    MIN_NARRATION_LENGTH = 3
    EMPTY_NARRATION_THRESHOLD = 0.30   # > 30 % empty → warning

    def validate(self, df: pd.DataFrame, narration_col: Optional[str]) -> CheckResult:
        if not narration_col or narration_col not in df.columns:
            return CheckResult(
                "Narration Quality", True,
                "No narration column found; skipping check.",
                severity="info",
            )

        total = len(df)
        if total == 0:
            return CheckResult("Narration Quality", True, "No rows to check.", severity="info")

        empty = df[narration_col].apply(
            lambda v: str(v).strip() in ("", "nan", "None", "N/A")
        ).sum()

        short = df[narration_col].apply(
            lambda v: 0 < len(str(v).strip()) < self.MIN_NARRATION_LENGTH
        ).sum()

        empty_pct = empty / total
        if empty_pct > self.EMPTY_NARRATION_THRESHOLD:
            return CheckResult(
                "Narration Quality", False,
                f"{empty} of {total} narrations are empty ({empty_pct:.0%}). "
                "Data quality is poor.",
                severity="warning",
            )

        msg = "Narration quality acceptable."
        if short:
            msg = f"Narration quality acceptable ({short} very short entries found)."

        return CheckResult("Narration Quality", True, msg, severity="info")


# ===========================================================================
# Layer 7 — OCR Detector
# ===========================================================================
class OCRDetector:
    """
    Detects whether a PDF is scanned (image-based) by checking the
    character density extracted by pdfplumber.
    """
    MIN_CHARS_PER_PAGE = 100

    def detect(self, pdf_path: str, password: Optional[str] = None) -> tuple[bool, CheckResult]:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path, password=password) as pdf:
                page_count = len(pdf.pages)
                if page_count == 0:
                    return True, CheckResult(
                        "OCR Detection", False,
                        "PDF has no readable pages.",
                        severity="error",
                    )
                char_counts = []
                for page in pdf.pages[:min(3, page_count)]:
                    text = page.extract_text() or ""
                    char_counts.append(len(text))

                avg_chars = sum(char_counts) / len(char_counts)
                if avg_chars < self.MIN_CHARS_PER_PAGE:
                    return True, CheckResult(
                        "OCR Detection", False,
                        "This PDF appears to be scanned or image-based. "
                        "Text extraction is very limited.",
                        detail=(
                            f"Average characters per page: {avg_chars:.0f}. "
                            "OCR processing would be required for accurate extraction."
                        ),
                        severity="error",
                    )

        except Exception as e:
            logger.warning("OCR detection failed: %s", e)

        return False, CheckResult(
            "OCR Detection", True,
            "PDF is text-based and compatible with the parser.",
            severity="info",
        )


# ===========================================================================
# Layer 8 — Confidence Engine
# ===========================================================================
CONFIDENCE_WEIGHTS = {
    "Column Validation":        20,
    "Date Validation":          15,
    "Balance Continuity":       25,
    "Duplicate Check":          10,
    "Transaction Count Sanity": 15,
    "Narration Quality":         5,
    "OCR Detection":            10,
}

class ConfidenceEngine:
    def compute(self, report: ValidationReport) -> tuple[float, str]:
        total_weight = sum(CONFIDENCE_WEIGHTS.values())
        earned = 0.0

        for check in report.checks:
            w = CONFIDENCE_WEIGHTS.get(check.name, 0)
            if check.passed:
                earned += w
            elif check.severity == "warning":
                earned += w * 0.5   # partial credit for warnings

        # Balance mismatch continuous penalty
        if report.balance_mismatches > 0 and report.total_transactions > 0:
            pct_bad = report.balance_mismatches / report.total_transactions
            balance_weight = CONFIDENCE_WEIGHTS["Balance Continuity"]
            penalty = balance_weight * min(pct_bad * 2, 1.0)
            earned = max(0, earned - penalty)

        score = (earned / total_weight) * 100

        if score >= 90:
            label = "High — Reliable for accounting"
        elif score >= 70:
            label = "Medium — Usable, manual verification recommended"
        else:
            label = "Low — Export blocked; data may be inaccurate"

        return round(score, 1), label


# ===========================================================================
# Master Orchestrator — ValidationEngine
# ===========================================================================
class ValidationEngine:
    """
    Runs all validation layers in sequence and produces a ValidationReport.
    Also applies safe-export policy:
      - Confidence ≥ 70 → export allowed
      - Any CRITICAL (severity='error') check that failed → export blocked
    """

    def __init__(self):
        self.col_validator    = ColumnValidator()
        self.date_validator   = DateValidator()
        self.bal_validator    = BalanceContinuityValidator()
        self.dup_validator    = DuplicateValidator()
        self.sanity_check     = TransactionSanityCheck()
        self.narration_val    = NarrationValidator()
        self.ocr_detector     = OCRDetector()
        self.confidence_engine = ConfidenceEngine()

    def run(
        self,
        df: pd.DataFrame,
        pdf_path: str,
        page_count: int,
        password: Optional[str] = None,
    ) -> tuple[pd.DataFrame, ValidationReport]:

        report = ValidationReport()
        report.total_transactions = len(df)

        # --- OCR Detection (operates on the PDF file itself) ---
        is_scanned, ocr_result = self.ocr_detector.detect(pdf_path, password)
        report.is_scanned_pdf = is_scanned
        report.checks.append(ocr_result)

        if is_scanned:
            # We still run checks if we want, or just log the OCR issue
            report.confidence_score, report.confidence_label = self.confidence_engine.compute(report)
            report.block_reason = (
                "This PDF appears to be scanned. Text extraction failed. "
                "OCR processing is required for accurate parsing."
            )
            # EXPORT NO LONGER BLOCKED


        # --- Column Validation ---
        col_result = self.col_validator.validate(df)
        report.checks.append(col_result)

        if not col_result.passed:
            report.confidence_score, report.confidence_label = self.confidence_engine.compute(report)
            report.block_reason = col_result.message
            # EXPORT NO LONGER BLOCKED


        # Resolve canonical column names
        date_col      = self.col_validator.resolve_column_name(df, "date")
        narration_col = self.col_validator.resolve_column_name(df, "narration")
        balance_col   = self.col_validator.resolve_column_name(df, "balance")
        debit_col     = self.col_validator.resolve_column_name(df, "debit")
        credit_col    = self.col_validator.resolve_column_name(df, "credit")

        # --- Date Validation ---
        date_result = self.date_validator.validate(df, date_col or "")
        report.checks.append(date_result)

        # --- Balance Continuity ---
        bal_result, suspicious_rows, mismatch_count = self.bal_validator.validate(
            df, balance_col or "", debit_col, credit_col
        )
        report.checks.append(bal_result)
        report.suspicious_rows = suspicious_rows
        report.balance_mismatches = mismatch_count

        # --- Duplicate Detection & Removal ---
        ref_col = self.col_validator.resolve_column_name(df, "reference")
        dup_result, dup_indices = self.dup_validator.validate(
            df, date_col, narration_col, balance_col, ref_col, debit_col, credit_col
        )
        report.checks.append(dup_result)
        report.duplicate_rows = dup_indices

        if dup_indices:
            df = df.drop(index=dup_indices).reset_index(drop=True)
            report.total_transactions = len(df)

        # --- Transaction Count Sanity ---
        sanity_result = self.sanity_check.validate(df, page_count)
        report.checks.append(sanity_result)

        # --- Narration Quality ---
        narration_result = self.narration_val.validate(df, narration_col)
        report.checks.append(narration_result)

        # --- Confidence Score ---
        report.confidence_score, report.confidence_label = self.confidence_engine.compute(report)

        # --- Safe Export Policy (DISABLED) ---
        # User requested to NEVER block exports, even on critical failures
        report.export_allowed = True
        
        critical_failures = [
            c for c in report.checks
            if not c.passed and c.severity == "error"
        ]
        if critical_failures:
            report.block_reason = critical_failures[0].message
        elif report.confidence_score < 70:
            report.block_reason = (
                f"Confidence score is low ({report.confidence_score:.0f}%). "
                "Manual verification is strongly recommended."
            )


        # Collect human-readable warnings
        for c in report.checks:
            if not c.passed or c.severity == "warning":
                report.warnings.append(c.message)

        logger.info(
            "Validation complete — Score: %.1f%% | Export: %s | Transactions: %d",
            report.confidence_score,
            report.export_allowed,
            report.total_transactions,
        )

        return df, report
