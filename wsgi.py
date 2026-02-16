import sys
import traceback

print("🚀 Gunicorn: Starting AgriPredictor-AI WSGI Server...")
try:
    from app import app
    print("✅ Flask app imported successfully")
except Exception as e:
    print(f"❌ CRITICAL ERROR importing app:")
    print(traceback.format_exc())
    sys.exit(1)

if __name__ == "__main__":
    app.run()
