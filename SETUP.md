# Setup Guide
 
## Prerequisites
 
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)
 
## Installation
 
### 1. Clone the Repository
 
```bash
git clone https://github.com/yourusername/Store-Sales---Time-Series-Forecasting.git
cd Store-Sales---Time-Series-Forecasting
```
 
### 2. Create Virtual Environment
 
**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```
 
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
 
### 3. Install Dependencies
 
```bash
pip install -r requirements.txt
```
 
## Running the Model
 
Execute the main forecasting pipeline:
 
```bash
python main.py
```
 
### Expected Output
 
The pipeline will:
1. Load training and test datasets from `data/` directory
2. Engineer 109 time-based features
3. Train a linear regression model
4. Generate predictions for 16-day test period
5. Save outputs:
   - `submission_notebook_method.csv` (28,512 predictions)
   - `time_series_model.pkl` (trained model)
 
### Performance Indicators
 
- Training RMSLE: ~0.55333
- Prediction coverage: 95.9% non-zero
- Processing time: ~30-60 seconds (depending on hardware)
 
## Data Requirements
 
Ensure the following CSV files are present in the `data/` directory:
- `train.csv` - Training sales data
- `test.csv` - Test data for predictions
- `holidays_events.csv` - Holiday calendar
- `stores.csv` - Store metadata (optional)
- `oil.csv` - Oil prices (optional)
- `transactions.csv` - Transaction counts (optional)
 
## Troubleshooting
 
### Missing Data Files
If you encounter "FileNotFoundError", ensure all required CSV files are in the `data/` directory.
 
### Memory Issues
The model uses categorical dtypes and float32 for memory efficiency. If issues persist, ensure at least 4GB RAM is available.
 
### Import Errors
Reinstall dependencies:
```bash
pip install -r requirements.txt --force-reinstall
```