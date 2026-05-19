@echo off
set PYTHON=python
set SCRIPTS_DIR=scripts

:: Default command prefix (runs via Docker)
set RUN=docker-compose exec pipeline

:: Check if "local" was passed as the second argument (e.g., make generate local)
if /I "%2"=="local" (
    set RUN=
    set EXTRA_ARGS=%3 %4 %5 %6
) else (
    set EXTRA_ARGS=%2 %3 %4 %5 %6
)

:: Direct Execution Routing
if "%1"=="" goto menu
if "%1"=="all" goto all
if "%1"=="generate" goto generate
if "%1"=="features" goto features
if "%1"=="evaluate" goto evaluate
if "%1"=="thresholds" goto thresholds
if "%1"=="validate" goto validate
if "%1"=="search" goto search
if "%1"=="test" goto test
if "%1"=="clean" goto clean
goto menu

:menu
echo ============================================================
echo           CLINICAL PIPELINE MANAGER (Windows)
echo ============================================================
echo   1. all        - Run entire pipeline
echo   2. generate   - Step 1: Generate Synthetic iCARE Data
echo   3. features   - Step 2: Build Phenotypes (SIRS/Pitt)
echo   4. evaluate   - Step 3: Compute Clinical Scores
echo   5. thresholds - Step 4: Evaluate Stewardship Thresholds
echo   6. validate   - Step 5: Run Clinical Audit (cases.csv)
echo   7. search     - Discover Clinical Codes (Keywords)
echo   8. test       - Run Pytest Suite
echo   9. clean      - Remove old reports
echo   0. exit       - Close the manager
echo ============================================================
echo NOTE: Commands run in Docker by default.
echo       To run locally, type: make.bat generate local
echo ============================================================
set /p choice="Enter choice (0-9): "

if "%choice%"=="1" goto all
if "%choice%"=="2" goto generate
if "%choice%"=="3" goto features
if "%choice%"=="4" goto evaluate
if "%choice%"=="5" goto thresholds
if "%choice%"=="6" goto validate
if "%choice%"=="7" goto search
if "%choice%"=="8" goto test
if "%choice%"=="9" goto clean
if "%choice%"=="0" goto :eof
goto menu

:all
call :generate
call :features
call :evaluate
call :thresholds
call :validate
goto :eof

:generate
echo.
echo --- Step 1: Generating Synthetic iCARE Data ---
%RUN% %PYTHON% -m %SCRIPTS_DIR%.01_generate_data %EXTRA_ARGS%
goto :eof

:features
echo.
echo --- Step 2: Building Clinical Features ---
%RUN% %PYTHON% -m %SCRIPTS_DIR%.02_build_features_icare %EXTRA_ARGS%
goto :eof

:evaluate
echo.
echo --- Step 3: Evaluating Clinical Scores ---
%RUN% %PYTHON% -m %SCRIPTS_DIR%.03_evaluate_scores %EXTRA_ARGS%
goto :eof

:thresholds
echo.
echo --- Step 4: Evaluating Stewardship Thresholds ---
%RUN% %PYTHON% -m %SCRIPTS_DIR%.04_evaluate_thresholds %EXTRA_ARGS%
goto :eof

:validate
echo.
echo --- Step 5: Validating Scores (Clinical Audit) ---
%RUN% %PYTHON% -m %SCRIPTS_DIR%.05_validate_scores %EXTRA_ARGS%
timeout /t 2 >nul
goto :eof

:test
echo.
echo --- Running Unit Tests ---
%RUN% %PYTHON% -m pytest tests/ %EXTRA_ARGS%
timeout /t 2 >nul
goto :eof

:search
echo.
echo --- Discover Clinical Codes (Keywords) ---
%RUN% %PYTHON% -m %SCRIPTS_DIR%.06_find_clinical_codes %EXTRA_ARGS%
timeout /t 2 >nul
goto :eof

:clean
echo.
echo --- 🧹 Cleaning up reports ---
if exist reports\*.log del /q reports\*.log
if exist reports\*.txt del /q reports\*.txt
echo Done.
timeout /t 2 >nul
goto :eof