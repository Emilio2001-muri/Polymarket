@echo off
title PolyBot - Trading Engine
echo ============================================
echo   PolyBot - Quantitative Trader
echo   Dashboard: http://localhost:8501
echo ============================================
echo.
echo Iniciando... (no cierres esta ventana)
echo.
cd /d "%~dp0"
start http://localhost:8501
streamlit run app.py --server.port 8501 --server.headless true
pause
