@echo off
REM Minecraft Hand Detector launcher: creates .venv on first run, installs pinned deps,
REM then starts the app. No global installs; everything stays in this folder.
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

REM --- 1. Virtual environment ---
if not exist "%PY%" (
    echo Creating virtual environment...
    py -3.13 -m venv .venv 2>nul || python -m venv .venv
    if not exist "%PY%" (
        echo.
        echo ERROR: Failed to create .venv. Install Python 3.13 ^(or make
        echo "python" available on PATH^), then delete .venv and re-run.
        pause
        exit /b 1
    )
)

REM --- 2. Pinned dependency check ---
"%PY%" -c "import importlib.metadata as m; assert m.version('mediapipe')=='1.0.1', m.version('mediapipe'); assert m.version('opencv-contrib-python')=='5.0.0.93', m.version('opencv-contrib-python'); assert m.version('cv2-enumerate-cameras')=='1.4.0', m.version('cv2-enumerate-cameras'); assert 'opencv-python' not in {d.metadata['Name'].lower() for d in m.distributions()}, 'stray opencv-python'" 2>nul
if errorlevel 1 (
    echo Installing pinned dependencies...
    "%PY%" -m pip uninstall -y opencv-python >nul 2>&1
    "%PY%" -m pip install --force-reinstall --no-deps opencv-contrib-python==5.0.0.93 || goto :install_fail
    "%PY%" -m pip install -r requirements.txt || goto :install_fail
    "%PY%" -c "import importlib.metadata as m; assert m.version('mediapipe')=='1.0.1'; assert m.version('opencv-contrib-python')=='5.0.0.93'; assert m.version('cv2-enumerate-cameras')=='1.4.0'; assert 'opencv-python' not in {d.metadata['Name'].lower() for d in m.distributions()}, 'stray opencv-python'" 2>nul || goto :install_fail
    echo Dependencies OK.
)

REM --- 3. Import sanity check (no camera/hardware access) ---
"%PY%" -c "import cv2, mediapipe, cv2_enumerate_cameras; assert cv2.__version__.startswith('5.0'), cv2.__version__"
if errorlevel 1 (
    echo.
    echo ERROR: Packages are installed but cv2/mediapipe fail to import
    echo correctly. Delete the .venv folder and re-run this script.
    pause
    exit /b 1
)

REM --- 4. Launch ---
"%PY%" main.py
if errorlevel 1 (
    echo.
    echo ERROR: Minecraft Hand Detector exited with an error. Check the output above.
    pause
    exit /b 1
)
exit /b 0

:install_fail
echo.
echo ERROR: Dependency install failed. Check your internet connection and
echo that requirements.txt pins are available for your Python version,
echo then re-run this script.
pause
exit /b 1
