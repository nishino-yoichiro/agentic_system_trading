@echo off
echo Data Inspector - Price Data Analysis Tool
echo ========================================

if "%1"=="" (
    echo Usage: test_data.bat SYMBOL [ACTION]
    echo.
    echo Examples:
    echo   test_data.bat BTC display
    echo   test_data.bat BTC chart
    echo   test_data.bat BTC validate
    echo   test_data.bat BTC compare
    echo.
    echo Available symbols: BTC, ETH, AAPL, MSFT, etc.
    echo Available actions: display, chart, range, validate, compare
    pause
    exit /b 1
)

set SYMBOL=%1
set ACTION=%2

if "%ACTION%"=="" set ACTION=display

echo Analyzing %SYMBOL% with action: %ACTION%
echo.

python tests/data_inspector.py --symbol %SYMBOL% --action %ACTION%

pause

