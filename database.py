import firebase_admin
from firebase_admin import credentials, firestore
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "firebase-admin-key.json")

db_client = None

try:
    if os.path.exists(DB_PATH):
        cred = credentials.Certificate(DB_PATH)
        firebase_admin.initialize_app(cred)
        db_client = firestore.client()
        print("Firebase Admin SDK initialized successfully.")
    else:
        print("WARNING: firebase-admin-key.json not found! Analytics won't be saved.")
except Exception as e:
    print(f"Error initializing Firebase Admin SDK: {e}")

def get_db():
    return db_client

def log_conversion(bank, status, file_size=0, user_email="guest", is_scanned=False, processing_time_ms=0):
    if not db_client: return
    try:
        db_client.collection("conversions").add({
            "bank": bank,
            "status": status,
            "file_size": file_size,
            "user_email": user_email,
            "is_scanned": is_scanned,
            "processing_time_ms": processing_time_ms,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"Firestore Error: {e}")

def log_error(bank, error_message, user_email="guest"):
    if not db_client: return
    try:
        db_client.collection("errors").add({
            "bank": bank,
            "error_message": error_message,
            "user_email": user_email,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"Firestore Error: {e}")

def log_user_login(email):
    if not db_client: return
    try:
        db_client.collection("users").document(email).set({
            "last_login": firestore.SERVER_TIMESTAMP,
            "email": email
        }, merge=True)
    except Exception as e:
        print(f"Firestore Error: {e}")
