@echo off
echo ================================================================
echo   Alzheimer MRI Dataset Inspection
echo ================================================================
echo.
python inspect_dataset.py --dataset_dir data\dataset %*
echo.
pause
