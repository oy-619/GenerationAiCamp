# PowerShell script to run uma3.py from root directory

Write-Host "Starting uma3.py from root directory..." -ForegroundColor Green
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the correct directory
if (-not (Test-Path "GenerationAiCamp.code-workspace")) {
    Write-Host "Warning: This script should be run from the GenerationAiCamp root directory" -ForegroundColor Red
    Write-Host "Current location: $(Get-Location)" -ForegroundColor Red
    Write-Host "Expected location: C:\work\ws_python\GenerationAiCamp" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Set Python path
$env:PYTHONPATH = "$(Get-Location)\Lesson25\uma3soft-app\src;$env:PYTHONPATH"

Write-Host "Starting uma3.py..." -ForegroundColor Yellow
Write-Host "Target script: $(Get-Location)\Lesson25\uma3soft-app\src\uma3.py" -ForegroundColor Cyan
Write-Host "Working directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host ""

try {
    # Run uma3.py
    python "Lesson25\uma3soft-app\src\uma3.py"
}
catch {
    Write-Host "Error running uma3.py: $_" -ForegroundColor Red
}
finally {
    Write-Host ""
    Write-Host "uma3.py finished." -ForegroundColor Green
    Read-Host "Press Enter to exit"
}
