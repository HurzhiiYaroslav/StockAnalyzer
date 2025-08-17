@echo off
echo Starting the application...

:: Start backend server
start cmd /k "cd backend && ..\venv_tf\Scripts\activate && set PYTHONPATH=%PYTHONPATH%;%CD% && python app.py"

:: Wait for backend to start
timeout /t 5

:: Start frontend server
start cmd /k "cd frontend && npm run dev"

echo Application started! Backend is running on http://localhost:5000
echo Frontend is running on http://localhost:5173 