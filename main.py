import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data in the same format as the notebook"""
    print("Loading datasets...")
    
    comp_dir = Path('data')  # Updated to use data directory
    
    # Load training data with proper dtypes
    train = pd.read_csv(comp_dir / 'train.csv', 
                       usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
                       dtype={
                           'store_nbr': 'category',
                           'family': 'category', 
                           'sales': 'float32',
                           'onpromotion': 'uint32'
                       },
                       parse_dates=['date'])
    
    # Convert to period format
    train['date'] = train.date.dt.to_period('D')
    train = train.set_index(['date', 'family', 'store_nbr']).sort_index()
    
    # Load test data
    test = pd.read_csv(comp_dir / 'test.csv', 
                      dtype={
                          'store_nbr': 'category',
                          'family': 'category',
                          'onpromotion': 'uint32'
                      },
                      parse_dates=['date'])
    
    test['date'] = test.date.dt.to_period('D')
    test = test.set_index(['date', 'family', 'store_nbr']).sort_index()
    
    # Load holiday events
    holiday_events = pd.read_csv(comp_dir / 'holidays_events.csv', 
                                dtype={
                                    'type': 'category',
                                    'locale': 'category', 
                                    'locale_name': 'category',
                                    'description': 'category',
                                    'transferred': 'bool'
                                },
                                parse_dates=['date'])
    holiday_events = holiday_events.set_index('date').to_period('D')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Training period: {train.index.get_level_values('date').min()} to {train.index.get_level_values('date').max()}")
    print(f"Test period: {test.index.get_level_values('date').min()} to {test.index.get_level_values('date').max()}")
    
    return train, test, holiday_events

def create_time_features(train_data, holiday_events):
    """Create sophisticated time-based features following the notebook approach"""
    
    # Create training target matrix (wide format)
    y_train = train_data.unstack(['family', 'store_nbr']).loc[:, 'sales']
    
    # Get unique dates for feature engineering
    index_ = train_data.index.get_level_values('date').unique()
    
    print(f"Creating features for {len(index_)} time periods")
    
    # 1. Basic time trend features
    dp = DeterministicProcess(
        index=index_,
        constant=True,
        order=1,  # Linear trend
        drop=True
    )
    X_features = dp.in_sample()
    
    # 2. Day of week seasonality
    day_of_week = pd.Series(X_features.index.dayofweek, index=index_)
    X_day_of_week = pd.get_dummies(day_of_week, prefix='day_of_week')
    X_features = pd.concat([X_features, X_day_of_week], axis=1)
    
    # 3. Annual Fourier features for capturing complex seasonality
    fourier = CalendarFourier(freq="A", order=10)
    dp_fourier = DeterministicProcess(
        index=index_,
        constant=False,
        order=1,
        seasonal=True,
        additional_terms=[fourier],
        drop=True
    )
    X_fourier = dp_fourier.in_sample()
    X_features = pd.concat([X_features, X_fourier], axis=1)
    
    # 4. Holiday features
    holidays = (holiday_events
                .query("transferred == False")  # Drop transferred holidays
                .query("locale == 'National'")  # Keep only national holidays
                .loc[:, 'description']
                .to_frame()
                .assign(description=lambda x: x.description.cat.remove_unused_categories())
               )
    
    # Remove duplicate dates
    duplicated_dates = holidays.index.duplicated(keep='first')
    holidays = holidays[~duplicated_dates]
    
    X_holidays = pd.get_dummies(holidays)
    X_features = X_features.join(X_holidays, on='date', how='left').fillna(0.0)
    
    print(f"Created {X_features.shape[1]} features")
    
    return X_features, y_train, dp, dp_fourier, X_holidays

def create_test_features(test_data, dp, dp_fourier, X_holidays):
    """Create features for test data using same transformations"""
    
    # Determine number of test periods
    test_dates = test_data.index.get_level_values('date').unique()
    n_test_periods = len(test_dates)
    
    print(f"Creating test features for {n_test_periods} periods")
    
    # 1. Basic time features
    X_test = dp.out_of_sample(steps=n_test_periods)
    X_test.index.name = 'date'
    
    # 2. Day of week features
    day_of_week = pd.Series(X_test.index.dayofweek, index=X_test.index)
    X_day_of_week = pd.get_dummies(day_of_week, prefix='day_of_week')
    X_test = pd.concat([X_test, X_day_of_week], axis=1)
    
    # 3. Fourier features
    X_fourier_test = dp_fourier.out_of_sample(n_test_periods)
    X_test = pd.concat([X_test, X_fourier_test], axis=1)
    
    # 4. Holiday features
    X_test = X_test.join(X_holidays, how='left').fillna(0.0)
    
    return X_test

def train_model(X_features, y_train):
    """Train the linear regression model"""
    
    print("Training model...")
    
    # Use Linear Regression as in the notebook
    model = LinearRegression(fit_intercept=False)
    model.fit(X_features, y_train)
    
    # Calculate training score
    y_pred_train = model.predict(X_features)
    y_pred_train = np.clip(y_pred_train, 0.0, None)  # Ensure non-negative
    
    # Calculate RMSLE on a sample for validation
    try:
        # Take a sample for RMSLE calculation (full calculation is expensive)
        sample_mask = np.random.choice(len(y_train), size=min(10000, len(y_train)), replace=False)
        y_sample = y_train.iloc[sample_mask].values.flatten()
        y_pred_sample = y_pred_train[sample_mask].flatten()
        
        # Remove any non-positive values for RMSLE calculation
        valid_mask = (y_sample > 0) & (y_pred_sample > 0)
        if valid_mask.sum() > 0:
            rmsle = np.sqrt(mean_squared_log_error(y_sample[valid_mask], y_pred_sample[valid_mask]))
            print(f"Training RMSLE (sample): {rmsle:.5f}")
    except Exception as e:
        print(f"Could not calculate RMSLE: {e}")
    
    return model

def make_predictions(model, X_test, test_data):
    """Make predictions and format for submission"""
    
    print("Making predictions...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, None)  # Ensure non-negative
    
    print(f"Prediction shape: {y_pred.shape}")
    
    # The model was trained on wide format with columns for each store-family combination
    # We need to recreate the same column structure as the training data
    
    # Get the original training structure to match columns
    # Load training data again to get the correct structure
    comp_dir = Path('data')  # Updated to use data directory
    train_orig = pd.read_csv(comp_dir / 'train.csv', 
                           usecols=['store_nbr', 'family', 'date', 'sales'],
                           dtype={'store_nbr': 'category', 'family': 'category'},
                           parse_dates=['date'])
    train_orig['date'] = train_orig.date.dt.to_period('D')
    train_orig = train_orig.set_index(['date', 'family', 'store_nbr']).sort_index()
    
    # Create the wide format structure to get correct columns
    y_train_structure = train_orig.unstack(['family', 'store_nbr']).loc[:, 'sales']
    
    # Create prediction DataFrame with correct columns
    y_pred_df = pd.DataFrame(y_pred, 
                            index=X_test.index,
                            columns=y_train_structure.columns)
    
    print(f"Prediction DataFrame shape: {y_pred_df.shape}")
    
    # Stack back to long format to match test data structure
    y_pred_stacked = y_pred_df.stack(['family', 'store_nbr']).to_frame()
    y_pred_stacked.columns = ['sales']
    # Reset index to get date, family, store_nbr as columns
    y_pred_stacked = y_pred_stacked.reset_index()
    
    # Fix column name after reset_index - the first level becomes 'level_0' instead of 'date'
    if 'level_0' in y_pred_stacked.columns:
        y_pred_stacked = y_pred_stacked.rename(columns={'level_0': 'date'})
    
    print(f"Stacked predictions shape: {y_pred_stacked.shape}")
    print(f"Stacked predictions columns: {y_pred_stacked.columns.tolist()}")
    
    # Load test data with IDs
    test_with_ids = pd.read_csv(comp_dir / 'test.csv')
    test_with_ids['date'] = pd.to_datetime(test_with_ids['date']).dt.to_period('D')
    
    print(f"Test data columns: {test_with_ids.columns.tolist()}")
    print(f"Test data shape: {test_with_ids.shape}")
    
    # Fix data type mismatch for merge
    # Convert prediction store_nbr from category to int to match test data
    y_pred_stacked['store_nbr'] = y_pred_stacked['store_nbr'].astype(str).astype(int)
    
    # Merge predictions with test IDs
    submission = test_with_ids.merge(
        y_pred_stacked, 
        on=['date', 'family', 'store_nbr'], 
        how='left'
    )[['id', 'sales']]
    
    # Sort by ID and handle any missing values
    submission = submission.sort_values('id').reset_index(drop=True)
    submission['sales'] = submission['sales'].fillna(0.0)
    
    print(f"Created submission with {len(submission)} predictions")
    print(f"Prediction range: ${submission['sales'].min():.2f} - ${submission['sales'].max():.2f}")
    print(f"Average prediction: ${submission['sales'].mean():.2f}")
    
    return submission

def main():
    """Main execution function following notebook methodology"""
    
    # Load data
    train, test, holiday_events = load_and_prepare_data()
    
    # Create features
    X_features, y_train, dp, dp_fourier, X_holidays = create_time_features(train, holiday_events)
    
    # Train model
    model = train_model(X_features, y_train)
    
    # Create test features
    X_test = create_test_features(test, dp, dp_fourier, X_holidays)
    
    # Make predictions
    submission = make_predictions(model, X_test, test)
    
    # Save submission
    submission.to_csv('submission_notebook_method.csv', index=False)
    print("Submission saved as 'submission_notebook_method.csv'")
    
    # Save model
    joblib.dump(model, 'time_series_model.pkl')
    print("Model saved as 'time_series_model.pkl'")
    
    return model, submission

if __name__ == "__main__":
    model, submission = main()