@echo off
echo Cleaning all generated files...
del world_stocks_cleaned.csv 2>nul
del prepared_data.csv 2>nul
del train_data.csv 2>nul
del test_data.csv 2>nul
del best_model.pkl 2>nul
del *.png 2>nul
echo Cleanup complete!
pause