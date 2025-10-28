@echo off
echo Starting uma3.py from root directory...
echo Current directory: %CD%
echo.

REM Check if we're in the correct directory
if not exist "GenerationAiCamp.code-workspace" (
    echo Warning: This script should be run from the GenerationAiCamp root directory
    echo Current location: %CD%
    echo Expected location: C:\work\ws_python\GenerationAiCamp
    pause
    exit /b 1
)

REM Set Python path
set PYTHONPATH=%CD%\Lesson25\uma3soft-app\src;%PYTHONPATH%

echo Starting uma3.py...
echo Target script: %CD%\Lesson25\uma3soft-app\src\uma3.py
echo Working directory: %CD%
echo.

REM Run uma3.py
python "Lesson25\uma3soft-app\src\uma3.py"

echo.
echo uma3.py finished.
pause
