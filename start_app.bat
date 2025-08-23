@echo off
title Launcher

echo Starting application components...

echo Launching Backend Server...
start "Backend Server - Flask" cmd /k "call .\venv_tf\Scripts\activate && python -m backend.app"

timeout /t 3 > nul

echo Launching Frontend Server...
start "Frontend Server - Vite" cmd /k "cd frontend && npm run dev"

echo.
echo Two server windows have been launched. This window will now close.
timeout /t 2 > nul

exit