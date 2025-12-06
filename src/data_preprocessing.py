"""
Data Preprocessing Pipeline
Handles data loading, cleaning, feature engineering, and sequence creation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class EnergyDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for energy consumption data
    """

    def __init__(self, sequence_length=30):
        """
        Initialize preprocessor

        Args:
            sequence_length: Number of time steps for sequence creation
        """
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.state_encoder = LabelEncoder()
        self.type_encoder = LabelEncoder()
        self.feature_names = None

    def load_data(self, file_path):
        """
        Load energy consumption data from CSV

        Args:
            file_path: Path to CSV file

        Returns:
            pandas DataFrame
        """
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records")
        return df

    def clean_data(self, df):
        """
        Clean and validate data

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        print("\nCleaning data...")
        initial_rows = len(df)

        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values
        print(f"Missing values:\n{df.isnull().sum()}")

        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)

        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # Remove outliers (values > 3 std dev)
        for col in ['Consumption', 'Energy_MU']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df = df[np.abs(df[col] - mean) <= (3 * std)]

        print(f"Removed {initial_rows - len(df)} rows during cleaning")
        print(f"Final dataset: {len(df)} records")

        return df

    def create_temporal_features(self, df, date_column='Date'):
        """
        Create time-based features

        Args:
            df: Input DataFrame
            date_column: Name of date column

        Returns:
            DataFrame with temporal features
        """
        print("\nCreating temporal features...")

        # Convert to datetime if not already
        df[date_column] = pd.to_datetime(df[date_column])

        # Extract temporal components
        df['Year'] = df[date_column].dt.year
        df['Month'] = df[date_column].dt.month
        df['Day'] = df[date_column].dt.day
        df['DayOfWeek'] = df[date_column].dt.dayofweek
        df['Quarter'] = df[date_column].dt.quarter
        df['DayOfYear'] = df[date_column].dt.dayofyear
        df['WeekOfYear'] = df[date_column].dt.isocalendar().week

        # Create cyclical features for better representation
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Day_Sin'] = np.sin(2 * np.pi * df['Day'] / 31)
        df['Day_Cos'] = np.cos(2 * np.pi * df['Day'] / 31)
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

        # Is weekend
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

        # Season (India specific: Winter, Summer, Monsoon)
        def get_season(month):
            if month in [12, 1, 2]:
                return 0  # Winter
            elif month in [3, 4, 5]:
                return 1  # Summer
            elif month in [6, 7, 8, 9]:
                return 2  # Monsoon
            else:
                return 3  # Post-Monsoon

        df['Season'] = df['Month'].apply(get_season)

        # Indian festivals and holidays (simplified)
        df['IsHoliday'] = 0  # Can be enhanced with actual holiday dates

        print(
            f"Created {sum([col.endswith(('_Sin', '_Cos', 'IsWeekend', 'Season', 'IsHoliday')) for col in df.columns])} temporal features")

        return df

    def create_lag_features(self, df, target_col='Consumption', lags=[1, 2, 3, 7, 14, 30]):
        """
        Create lagged features for time series

        Args:
            df: Input DataFrame
            target_col: Target column name
            lags: List of lag periods

        Returns:
            DataFrame with lag features
        """
        print(f"\nCreating lag features for {target_col}...")

        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)

        # Create lag features
        for lag in lags:
            df[f'{target_col}_Lag_{lag}'] = df[target_col].shift(lag)

        print(f"Created {len(lags)} lag features")

        return df

    def create_rolling_features(self, df, target_col='Consumption', windows=[7, 14, 30]):
        """
        Create rolling window statistics

        Args:
            df: Input DataFrame
            target_col: Target column name
            windows: List of window sizes

        Returns:
            DataFrame with rolling features
        """
        print(f"\nCreating rolling features for {target_col}...")

        for window in windows:
            df[f'{target_col}_RollingMean_{window}'] = df[target_col].rolling(
                window=window, min_periods=1
            ).mean()
            df[f'{target_col}_RollingStd_{window}'] = df[target_col].rolling(
                window=window, min_periods=1
            ).std()
            df[f'{target_col}_RollingMin_{window}'] = df[target_col].rolling(
                window=window, min_periods=1
            ).min()
            df[f'{target_col}_RollingMax_{window}'] = df[target_col].rolling(
                window=window, min_periods=1
            ).max()

        print(f"Created {len(windows) * 4} rolling features")

        return df

    def encode_categorical_features(self, df):
        """
        Encode categorical variables

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with encoded features
        """
        print("\nEncoding categorical features...")

        # Encode State
        if 'State' in df.columns:
            df['State_Encoded'] = self.state_encoder.fit_transform(df['State'])

        # Encode Consumption Type
        if 'Type' in df.columns or 'ConsumptionType' in df.columns:
            type_col = 'Type' if 'Type' in df.columns else 'ConsumptionType'
            df['Type_Encoded'] = self.type_encoder.fit_transform(df[type_col])

        print("Categorical encoding complete")

        return df

    def create_sequences(self, df, target_col='Consumption'):
        """
        Create sequences for LSTM input

        Args:
            df: Input DataFrame
            target_col: Target column name

        Returns:
            X (sequences), y (targets)
        """
        print(f"\nCreating sequences of length {self.sequence_length}...")

        # Select feature columns (exclude date and target)
        feature_cols = [col for col in df.columns if col not in
                        ['Date', target_col, 'State', 'Type', 'ConsumptionType']]

        self.feature_names = feature_cols

        # Extract features and target
        features = df[feature_cols].values
        target = df[target_col].values

        X, y = [], []

        # Create sequences
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:i + self.sequence_length])
            y.append(target[i + self.sequence_length])

        X = np.array(X)
        y = np.array(y)

        print(f"Created {len(X)} sequences")
        print(f"X shape: {X.shape}, y shape: {y.shape}")

        return X, y

    def scale_features(self, X_train, X_val, X_test):
        """
        Scale features using StandardScaler

        Args:
            X_train, X_val, X_test: Train, validation, test sets

        Returns:
            Scaled datasets
        """
        print("\nScaling features...")

        # Reshape for scaling
        n_samples_train, n_timesteps, n_features = X_train.shape
        n_samples_val = X_val.shape[0]
        n_samples_test = X_test.shape[0]

        X_train_reshaped = X_train.reshape(-1, n_features)
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)

        # Fit and transform
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_val_scaled = self.scaler.transform(X_val_reshaped)
        X_test_scaled = self.scaler.transform(X_test_reshaped)

        # Reshape back
        X_train_scaled = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
        X_val_scaled = X_val_scaled.reshape(n_samples_val, n_timesteps, n_features)
        X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)

        print("Feature scaling complete")

        return X_train_scaled, X_val_scaled, X_test_scaled

    def preprocess_pipeline(self, df, target_col='Consumption'):
        """
        Complete preprocessing pipeline

        Args:
            df: Input DataFrame
            target_col: Target column name

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("=" * 50)
        print("STARTING PREPROCESSING PIPELINE")
        print("=" * 50)

        # Clean data
        df = self.clean_data(df)

        # Create temporal features
        df = self.create_temporal_features(df)

        # Create lag features
        df = self.create_lag_features(df, target_col)

        # Create rolling features
        df = self.create_rolling_features(df, target_col)

        # Encode categorical features
        df = self.encode_categorical_features(df)

        # Remove NaN values created by lag/rolling features
        df = df.dropna().reset_index(drop=True)

        print(f"\nFinal preprocessed dataset: {len(df)} records")

        # Create sequences
        X, y = self.create_sequences(df, target_col)

        # Split data: 70% train, 15% validation, 15% test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.15, shuffle=False
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.176, shuffle=False  # 0.176 * 0.85 ≈ 0.15
        )

        # Scale features
        X_train, X_val, X_test = self.scale_features(X_train, X_val, X_test)

        print("\n" + "=" * 50)
        print("PREPROCESSING COMPLETE")
        print("=" * 50)
        print(f"Train set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Feature dimensions: {X_train.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """Save the preprocessor with scalers and encoders"""
        preprocessor_data = {
            'scaler': self.scaler,
            'state_encoder': self.state_encoder,
            'type_encoder': self.type_encoder,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length
        }

        with open(path, 'wb') as f:
            pickle.dump(preprocessor_data, f)

        print(f"\nPreprocessor saved to {path}")

    def load_preprocessor(self, path='models/preprocessor.pkl'):
        """Load a saved preprocessor"""
        with open(path, 'rb') as f:
            preprocessor_data = pickle.load(f)

        self.scaler = preprocessor_data['scaler']
        self.state_encoder = preprocessor_data['state_encoder']
        self.type_encoder = preprocessor_data['type_encoder']
        self.feature_names = preprocessor_data['feature_names']
        self.sequence_length = preprocessor_data['sequence_length']

        print(f"Preprocessor loaded from {path}")


# Example usage
if __name__ == "__main__":
    # Create synthetic dataset for demonstration
    print("Creating synthetic dataset for demonstration...")

    dates = pd.date_range(start='2020-01-01', end='2025-11-30', freq='D')
    states = ['Maharashtra', 'Gujarat', 'Karnataka', 'Tamil Nadu', 'UP']
    types = ['Residential', 'Industrial', 'Agricultural', 'Commercial']

    np.random.seed(42)

    data = []
    for date in dates:
        for state in states:
            for cons_type in types:
                # Base consumption with trend and seasonality
                base = np.random.randint(1000, 5000)
                trend = (date - dates[0]).days * 0.5
                seasonal = 500 * np.sin(2 * np.pi * date.month / 12)
                noise = np.random.normal(0, 100)

                consumption = base + trend + seasonal + noise

                data.append({
                    'Date': date,
                    'State': state,
                    'Type': cons_type,
                    'Consumption': max(0, consumption)
                })

    df = pd.DataFrame(data)
    print(f"Generated {len(df)} records")

    # Save to CSV
    df.to_csv('data/raw/synthetic_energy_data.csv', index=False)

    # Initialize preprocessor
    preprocessor = EnergyDataPreprocessor(sequence_length=30)

    # Run preprocessing pipeline
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline(df)

    # Save preprocessor
    preprocessor.save_preprocessor()

    print("\nPreprocessing pipeline complete!")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val: X={X_val.shape}, y={y_val.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")