@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "DL_GPU=0"
set "DL_FAST=0"

echo.
echo ==============================================
echo   LAB 01 - kompletne rozwiazanie lokalne
echo ==============================================
echo.
if not exist ".venv\Scripts\python.exe" (
    echo Pierwszy start: tworze .venv i instaluje biblioteki.
    call "INSTALUJ_SRODOWISKO.bat"
    if errorlevel 1 goto error
)
if not exist "wyniki" mkdir "wyniki"

".venv\Scripts\python.exe" -u "lab01_rozwiazanie.py"
if errorlevel 1 goto error

echo.
echo ==============================================
echo SUKCES. Wyniki i wykresy sa w folderze wyniki.
echo ==============================================
if /i "%LAB_NO_PAUSE%"=="1" exit /b 0
pause
exit /b 0

:error
echo.
echo ==============================================
echo BLAD: program nie zakonczyl sie poprawnie.
echo Sprawdz komunikat wyzej. Niczego nie uznajemy za zaliczone.
echo ==============================================
if /i "%LAB_NO_PAUSE%"=="1" exit /b 1
pause
exit /b 1
