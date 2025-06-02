@echo off
echo [1/6] Cleaning generated files...
del world_stocks_cleaned.csv 2>nul
del prepared_data.csv 2>nul
del train_data.csv 2>nul
del test_data.csv 2>nul
del best_model.pkl 2>nul
del *.png 2>nul

echo [2/6] Extracting and cleaning data...
python extract_clean_data.py
if errorlevel 1 (
    echo Error in extraction step!
    pause
    exit /b
)

echo [3/6] Preparing data...
python data_preparation.py
if errorlevel 1 (
    echo Error in preparation step!
    pause
    exit /b
)

echo [4/6] Engineering features...
python feature_engineering.py
if errorlevel 1 (
    echo Error in feature engineering!
    pause
    exit /b
)

echo [5/6] Training models...
python model_training.py
if errorlevel 1 (
    echo Error in model training!
    pause
    exit /b
)

echo [6/6] Creating visualizations...
python visualize.py

echo.
echo Pipeline complete! Check results:
echo - Trained model: best_model.pkl
echo - Visualizations: *.png
echo.
echo Note: Run clean.bat to remove all generated files
pause