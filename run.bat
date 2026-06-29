@echo off
REM run.bat - Start the Enterprise Knowledge Assistant (Windows)

echo.
echo ╔══════════════════════════════════════════════╗
echo ║   Enterprise Knowledge Assistant - Startup  ║
echo ╚══════════════════════════════════════════════╝
echo.

cd backend

REM Check for .env
if not exist .env (
    if exist .env.example (
        copy .env.example .env
        echo ⚠  Created .env from template.
        echo    Open backend\.env and set your GEMINI_API_KEY
        echo    Then run this script again.
        pause
        exit /b 1
    )
)

REM Create venv if needed
if not exist venv (
    echo 🔧 Creating virtual environment...
    python -m venv venv
)

REM Activate and install
call venv\Scripts\activate.bat

echo 📦 Installing dependencies...
pip install -q -r requirements.txt

echo.
echo ✅ Setup complete!
echo 🚀 Starting backend on http://localhost:8000
echo 📚 API docs at http://localhost:8000/docs
echo.
echo    Open frontend\index.html in your browser to use the UI
echo    Press Ctrl+C to stop
echo.

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
