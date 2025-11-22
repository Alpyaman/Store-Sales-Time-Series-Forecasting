"""
Store Sales Time Series Forecasting
====================================
 
This module implements a sophisticated time series forecasting solution for predicting
store sales using linear regression with advanced temporal features including:
- Linear trends
- Day-of-week seasonality
- Annual Fourier components
- National holiday effects
 
Author: Store Sales Forecasting Team
License: MIT
"""
 
from pathlib import Path
from typing import Tuple
import warnings
 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
import joblib
 
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
 
class Config:
    """Configuration parameters for the forecasting model."""
 
    # Data paths
    DATA_DIR = Path('data')
    MODEL_OUTPUT_PATH = 'time_series_model.pkl'
    SUBMISSION_OUTPUT_PATH = 'submission_notebook_method.csv'
 
    # Data column configurations
    TRAIN_COLUMNS = ['store_nbr', 'family', 'date', 'sales', 'onpromotion']
    TEST_COLUMNS = ['store_nbr', 'family', 'date', 'onpromotion']
 
    # Data types for memory optimization
    DTYPE_CONFIG = {
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
        'onpromotion': 'uint32'
    }
 
    HOLIDAY_DTYPE_CONFIG = {
        'type': 'category',
        'locale': 'category',
        'locale_name': 'category',
        'description': 'category',
        'transferred': 'bool'
    }
 
    # Model parameters
    LINEAR_TREND_ORDER = 1  # Linear trend
    FOURIER_ORDER = 10  # Number of Fourier components for annual seasonality
    FOURIER_FREQUENCY = "A"  # Annual frequency
 
    # Validation parameters
    RMSLE_SAMPLE_SIZE = 10000  # Sample size for RMSLE calculation
    RANDOM_SEED = 42
 
    # Feature engineering
    HOLIDAY_LOCALE = 'National'  # Only use national holidays
    INCLUDE_TRANSFERRED_HOLIDAYS = False  # Exclude transferred holidays

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================
 
def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare training, test, and holiday data for time series forecasting.
 
    This function loads the competition datasets with optimized data types,
    converts dates to period format, and creates a multi-index structure
    (date, family, store_nbr) for efficient time series operations.
 
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train : Training data with MultiIndex (date, family, store_nbr)
                containing sales and promotion information
        test : Test data with same MultiIndex structure
        holiday_events : Holiday events data indexed by date period
 
    Notes
    -----
    - Uses categorical dtypes for memory efficiency
    - Converts dates to Period format for time series alignment
    - Data is sorted by index for optimal performance
    """
    print("Loading datasets...")
 
    # Load training data with memory-optimized dtypes
    train = pd.read_csv(
        Config.DATA_DIR / 'train.csv',
        usecols=Config.TRAIN_COLUMNS,
        dtype=Config.DTYPE_CONFIG,
        parse_dates=['date']
    )
 
    # Convert to period format for time series consistency
    train['date'] = train.date.dt.to_period('D')
    train = train.set_index(['date', 'family', 'store_nbr']).sort_index()
 
    # Load test data with same structure
    test = pd.read_csv(
        Config.DATA_DIR / 'test.csv',
        dtype={k: v for k, v in Config.DTYPE_CONFIG.items() if k != 'sales'},
        parse_dates=['date']
    )
 
    test['date'] = test.date.dt.to_period('D')
    test = test.set_index(['date', 'family', 'store_nbr']).sort_index()
 
    # Load holiday events data
    holiday_events = pd.read_csv(
        Config.DATA_DIR / 'holidays_events.csv',
        dtype=Config.HOLIDAY_DTYPE_CONFIG,
        parse_dates=['date']
    )
    holiday_events = holiday_events.set_index('date').to_period('D')
 
    # Print data summary
    _print_data_summary(train, test)
 
    return train, test, holiday_events
 
 
def _print_data_summary(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print summary statistics for loaded datasets."""
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Training period: {train.index.get_level_values('date').min()} "
          f"to {train.index.get_level_values('date').max()}")
    print(f"Test period: {test.index.get_level_values('date').min()} "
          f"to {test.index.get_level_values('date').max()}")
    
# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
 
def create_time_features(
    train_data: pd.DataFrame,
    holiday_events: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, DeterministicProcess, DeterministicProcess, pd.DataFrame]:
    """
    Create sophisticated time-based features for the forecasting model.
 
    Constructs a comprehensive feature set including:
    1. Linear time trends
    2. Day-of-week seasonality (7 dummy variables)
    3. Annual Fourier components (capturing yearly patterns)
    4. National holiday indicators
 
    Parameters
    ----------
    train_data : pd.DataFrame
        Training data with MultiIndex (date, family, store_nbr)
    holiday_events : pd.DataFrame
        Holiday events data indexed by date period
 
    Returns
    -------
    Tuple containing:
        X_features : Feature matrix for training (dates × features)
        y_train : Target matrix in wide format (dates × store-family combinations)
        dp : DeterministicProcess for basic time features
        dp_fourier : DeterministicProcess for Fourier features
        X_holidays : Holiday feature DataFrame for test data processing
 
    Notes
    -----
    - Wide format enables multi-target learning across all store-family pairs
    - Features are engineered to capture multiple seasonal scales
    """
    # Reshape target to wide format: dates as rows, (family, store) as columns
    y_train = train_data.unstack(['family', 'store_nbr']).loc[:, 'sales']
 
    # Extract unique dates for feature engineering
    date_index = train_data.index.get_level_values('date').unique()
 
    print(f"Creating features for {len(date_index)} time periods")
 
    # Build features progressively
    X_features = _create_trend_features(date_index)
    X_features = _add_day_of_week_features(X_features, date_index)
    dp_fourier, X_fourier = _create_fourier_features(date_index)
    X_features = pd.concat([X_features, X_fourier], axis=1)
 
    # Get deterministic process for later use
    dp = DeterministicProcess(
        index=date_index,
        constant=True,
        order=Config.LINEAR_TREND_ORDER,
        drop=True
    )
 
    # Add holiday features
    X_holidays = _prepare_holiday_features(holiday_events)
    X_features = X_features.join(X_holidays, on='date', how='left').fillna(0.0)
 
    print(f"Created {X_features.shape[1]} features")
 
    return X_features, y_train, dp, dp_fourier, X_holidays
 
 
def _create_trend_features(date_index: pd.PeriodIndex) -> pd.DataFrame:
    """Create linear trend features using DeterministicProcess."""
    dp = DeterministicProcess(
        index=date_index,
        constant=True,  # Include intercept
        order=Config.LINEAR_TREND_ORDER,  # Linear trend
        drop=True  # Drop redundant features
    )
    return dp.in_sample()
 
 
def _add_day_of_week_features(
    X_features: pd.DataFrame,
    date_index: pd.PeriodIndex
) -> pd.DataFrame:
    """Add day-of-week dummy variables for weekly seasonality."""
    day_of_week = pd.Series(X_features.index.dayofweek, index=date_index)
    X_day_of_week = pd.get_dummies(day_of_week, prefix='day_of_week')
    return pd.concat([X_features, X_day_of_week], axis=1)
 
 
def _create_fourier_features(
    date_index: pd.PeriodIndex
) -> Tuple[DeterministicProcess, pd.DataFrame]:
    """
    Create Fourier features for capturing annual seasonality.
 
    Fourier terms model smooth, periodic patterns that repeat annually.
    Higher order captures more complex seasonal patterns.
    """
    fourier = CalendarFourier(freq=Config.FOURIER_FREQUENCY, order=Config.FOURIER_ORDER)
    dp_fourier = DeterministicProcess(
        index=date_index,
        constant=False,
        order=1,
        seasonal=True,
        additional_terms=[fourier],
        drop=True
    )
    X_fourier = dp_fourier.in_sample()
    return dp_fourier, X_fourier
 
 
def _prepare_holiday_features(holiday_events: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare national holiday features.
 
    Filters to national holidays only, removes transferred holidays,
    and creates one-hot encoded features for each unique holiday.
    """
    # Filter holidays based on configuration
    query_str = f"transferred == {Config.INCLUDE_TRANSFERRED_HOLIDAYS}"
    holidays = (
        holiday_events
        .query(query_str)
        .query(f"locale == '{Config.HOLIDAY_LOCALE}'")
        .loc[:, 'description']
        .to_frame()
        .assign(description=lambda x: x.description.cat.remove_unused_categories())
    )
 
    # Remove duplicate dates (keep first occurrence)
    holidays = holidays[~holidays.index.duplicated(keep='first')]
 
    # One-hot encode holiday descriptions
    return pd.get_dummies(holidays)

def create_test_features(
    test_data: pd.DataFrame,
    dp: DeterministicProcess,
    dp_fourier: DeterministicProcess,
    X_holidays: pd.DataFrame
) -> pd.DataFrame:
    """
    Create features for test data using the same transformations as training.
 
    Applies the fitted DeterministicProcess objects to generate out-of-sample
    features for the test period, ensuring consistency with training features.
 
    Parameters
    ----------
    test_data : pd.DataFrame
        Test data with MultiIndex (date, family, store_nbr)
    dp : DeterministicProcess
        Fitted process for trend features
    dp_fourier : DeterministicProcess
        Fitted process for Fourier features
    X_holidays : pd.DataFrame
        Holiday features to join with test dates
 
    Returns
    -------
    pd.DataFrame
        Feature matrix for test period with same columns as training features
    """
    # Determine test period length
    test_dates = test_data.index.get_level_values('date').unique()
    n_test_periods = len(test_dates)
 
    print(f"Creating test features for {n_test_periods} periods")
 
    # Generate out-of-sample features using fitted processes
    X_test = dp.out_of_sample(steps=n_test_periods)
    X_test.index.name = 'date'
 
    # Add day-of-week features
    day_of_week = pd.Series(X_test.index.dayofweek, index=X_test.index)
    X_day_of_week = pd.get_dummies(day_of_week, prefix='day_of_week')
    X_test = pd.concat([X_test, X_day_of_week], axis=1)
 
    # Add Fourier features
    X_fourier_test = dp_fourier.out_of_sample(n_test_periods)
    X_test = pd.concat([X_test, X_fourier_test], axis=1)
 
    # Add holiday features (fill with 0 for non-holiday dates)
    X_test = X_test.join(X_holidays, how='left').fillna(0.0)
 
    return X_test

# =============================================================================
# MODEL TRAINING
# =============================================================================
 
def train_model(X_features: pd.DataFrame, y_train: pd.DataFrame) -> LinearRegression:
    """
    Train a linear regression model on the time series features.
 
    Uses LinearRegression without intercept (included in features via
    DeterministicProcess). The model learns separate coefficients for
    each store-family combination in the wide format target.
 
    Parameters
    ----------
    X_features : pd.DataFrame
        Feature matrix (n_dates × n_features)
    y_train : pd.DataFrame
        Target matrix in wide format (n_dates × n_store_family_combinations)
 
    Returns
    -------
    LinearRegression
        Fitted model ready for predictions
 
    Notes
    -----
    - fit_intercept=False because intercept is in feature matrix
    - Predictions are clipped to ensure non-negative sales
    - RMSLE is computed on a sample for efficiency
    """
    print("Training model...")
 
    # Initialize and train model
    model = LinearRegression(fit_intercept=False)
    model.fit(X_features, y_train)
 
    # Validate model performance on training data
    _validate_training_performance(model, X_features, y_train)
 
    return model
 
 
def _validate_training_performance(
    model: LinearRegression,
    X_features: pd.DataFrame,
    y_train: pd.DataFrame
) -> None:
    """
    Calculate and print training RMSLE on a sample.
 
    Uses sampling for computational efficiency on large datasets.
    """
    try:
        # Generate predictions and ensure non-negative values
        y_pred_train = model.predict(X_features)
        y_pred_train = np.clip(y_pred_train, 0.0, None)
 
        # Sample for efficient RMSLE calculation
        np.random.seed(Config.RANDOM_SEED)
        sample_size = min(Config.RMSLE_SAMPLE_SIZE, len(y_train))
        sample_indices = np.random.choice(len(y_train), size=sample_size, replace=False)
 
        y_sample = y_train.iloc[sample_indices].values.flatten()
        y_pred_sample = y_pred_train[sample_indices].flatten()
 
        # RMSLE requires positive values only
        valid_mask = (y_sample > 0) & (y_pred_sample > 0)
 
        if valid_mask.sum() > 0:
            rmsle = np.sqrt(mean_squared_log_error(
                y_sample[valid_mask],
                y_pred_sample[valid_mask]
            ))
            print(f"Training RMSLE (sample of {valid_mask.sum():,}): {rmsle:.5f}")
        else:
            print("Warning: No valid positive predictions for RMSLE calculation")
 
    except Exception as e:
        print(f"Could not calculate RMSLE: {e}")


# =============================================================================
# PREDICTION AND SUBMISSION GENERATION
# =============================================================================
 
def make_predictions(
    model: LinearRegression,
    X_test: pd.DataFrame,
    test_data: pd.DataFrame,
    y_train_columns: pd.MultiIndex
) -> pd.DataFrame:
    """
    Generate predictions and format them for Kaggle submission.
 
    This function handles the complex transformation from wide-format predictions
    back to the long-format submission file required by Kaggle.
 
    Parameters
    ----------
    model : LinearRegression
        Trained forecasting model
    X_test : pd.DataFrame
        Test features (n_test_dates × n_features)
    test_data : pd.DataFrame
        Original test data with MultiIndex for ID matching
    y_train_columns : pd.MultiIndex
        Column structure from training (family, store_nbr) for reshaping
 
    Returns
    -------
    pd.DataFrame
        Submission file with columns ['id', 'sales']
 
    Notes
    -----
    The transformation pipeline:
    1. Predict in wide format (dates × store-family combinations)
    2. Stack to long format (date, family, store_nbr)
    3. Convert data types for merge compatibility
    4. Merge with test IDs to create submission format
    """
    print("Making predictions...")
 
    # Generate predictions and ensure non-negative sales
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, None)
 
    print(f"Prediction shape: {y_pred.shape}")
 
    # Create DataFrame with same structure as training target
    y_pred_df = pd.DataFrame(
        y_pred,
        index=X_test.index,
        columns=y_train_columns
    )
 
    print(f"Prediction DataFrame shape: {y_pred_df.shape}")
 
    # Transform to long format for submission
    predictions_long = _reshape_predictions_to_long_format(y_pred_df)
 
    # Merge with test IDs to create submission
    submission = _create_submission_file(predictions_long, test_data)
 
    # Print summary statistics
    _print_prediction_summary(submission)
 
    return submission
 
 
def _reshape_predictions_to_long_format(y_pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape wide-format predictions to long format.
 
    Converts from (dates × store-family) to (date, family, store_nbr, sales).
    """
    # Stack from wide to long format
    predictions = y_pred_df.stack(['family', 'store_nbr']).to_frame()
    predictions.columns = ['sales']
    predictions = predictions.reset_index()
 
    # Fix column name if necessary (pandas may use 'level_0' for date)
    if 'level_0' in predictions.columns:
        predictions = predictions.rename(columns={'level_0': 'date'})
 
    # Convert store_nbr from category to int for merge compatibility
    predictions['store_nbr'] = predictions['store_nbr'].astype(str).astype(int)
 
    print(f"Reshaped predictions: {predictions.shape}")
    print(f"Columns: {predictions.columns.tolist()}")
 
    return predictions
 
 
def _create_submission_file(
    predictions: pd.DataFrame,
    test_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge predictions with test IDs to create submission file.
 
    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions in long format with (date, family, store_nbr, sales)
    test_data : pd.DataFrame
        Test data with IDs for merging
 
    Returns
    -------
    pd.DataFrame
        Submission DataFrame with ['id', 'sales'] sorted by ID
    """
    # Load test data with IDs
    test_with_ids = pd.read_csv(Config.DATA_DIR / 'test.csv')
    test_with_ids['date'] = pd.to_datetime(test_with_ids['date']).dt.to_period('D')
 
    print(f"Test data shape: {test_with_ids.shape}")
 
    # Merge predictions with test IDs
    submission = test_with_ids.merge(
        predictions,
        on=['date', 'family', 'store_nbr'],
        how='left'
    )[['id', 'sales']]
 
    # Sort by ID and handle any missing values
    submission = submission.sort_values('id').reset_index(drop=True)
    submission['sales'] = submission['sales'].fillna(0.0)
 
    return submission
 
 
def _print_prediction_summary(submission: pd.DataFrame) -> None:
    """Print summary statistics for the predictions."""
    print("\nSubmission Summary:")
    print(f"  Total predictions: {len(submission):,}")
    print(f"  Sales range: ${submission['sales'].min():.2f} - ${submission['sales'].max():.2f}")
    print(f"  Average sale: ${submission['sales'].mean():.2f}")
    print(f"  Median sale: ${submission['sales'].median():.2f}")
    print(f"  Non-zero predictions: {(submission['sales'] > 0).sum():,} "
          f"({(submission['sales'] > 0).sum() / len(submission) * 100:.1f}%)")
    
# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================
 
def main() -> Tuple[LinearRegression, pd.DataFrame]:
    """
    Main execution pipeline for store sales forecasting.
 
    This function orchestrates the complete machine learning workflow:
    1. Load and prepare data
    2. Engineer time series features
    3. Train linear regression model
    4. Generate predictions for test period
    5. Save model and submission files
 
    Returns
    -------
    Tuple[LinearRegression, pd.DataFrame]
        model : Trained forecasting model
        submission : Kaggle submission DataFrame
 
    Notes
    -----
    Output files:
    - submission_notebook_method.csv: Kaggle submission format
    - time_series_model.pkl: Serialized trained model
    """
    print("=" * 70)
    print("STORE SALES TIME SERIES FORECASTING")
    print("=" * 70)
 
    # Step 1: Load and prepare data
    print("\n[1/5] Loading data...")
    train, test, holiday_events = load_and_prepare_data()
 
    # Step 2: Create training features
    print("\n[2/5] Engineering features...")
    X_features, y_train, dp, dp_fourier, X_holidays = create_time_features(
        train, holiday_events
    )
 
    # Step 3: Train model
    print("\n[3/5] Training model...")
    model = train_model(X_features, y_train)
 
    # Step 4: Create test features
    print("\n[4/5] Creating test features...")
    X_test = create_test_features(test, dp, dp_fourier, X_holidays)
 
    # Step 5: Generate predictions
    print("\n[5/5] Generating predictions...")
    submission = make_predictions(model, X_test, test, y_train.columns)
 
    # Save outputs
    _save_outputs(model, submission)
 
    print("\n" + "=" * 70)
    print("FORECASTING COMPLETE!")
    print("=" * 70)
 
    return model, submission
 
 
def _save_outputs(model: LinearRegression, submission: pd.DataFrame) -> None:
    """Save model and submission files to disk."""
    # Save submission file
    submission.to_csv(Config.SUBMISSION_OUTPUT_PATH, index=False)
    print(f"\n✓ Submission saved: {Config.SUBMISSION_OUTPUT_PATH}")
 
    # Save trained model
    joblib.dump(model, Config.MODEL_OUTPUT_PATH)
    print(f"✓ Model saved: {Config.MODEL_OUTPUT_PATH}")
 
 
if __name__ == "__main__":
    model, submission = main()