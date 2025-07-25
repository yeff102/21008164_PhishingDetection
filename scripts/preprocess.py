import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class PhishingDataPreprocessor:
    """
    Comprehensive preprocessing pipeline for phishing detection dataset
    with mixed feature types and missing values from dataset merging.
    """
    
    def __init__(self):
        self.scaler = None
        self.feature_stats = {}
        
        # Define feature categories based on your dataset
        self.url_features = [
            'URLLength', 'DomainLength', 'TLD_known', 'URLSimilarityIndex', 
            'CharContinuationRate', 'TLDLegitimateProb', 'URLCharProb', 
            'NoOfSubDomain', 'HasObfuscation', 'ObfuscationRatio', 
            'LetterRatioInURL', 'SpacialCharRatioInURL', 'url_has_ip', 
            'domain_entropy'
        ]
        
        self.content_features = [
            'HasTitle', 'DomainTitleMatchScore', 'URLTitleMatchScore', 
            'HasDescription', 'HasSocialNet', 'HasSubmitButton', 
            'HasHiddenFields', 'HasPasswordField', 'HasCopyrightInfo', 
            'NoOfImage', 'NoOfCSS', 'NoOfJS', 'NoOfSelfRef', 'NoOfExternalRef'
        ]
        
        self.security_features = [
            'IsHTTPS', 'HasFavicon', 'Robots', 'IsResponsive'
        ]
        
        self.categorical_features = [
            'Bank', 'Pay', 'Crypto'
        ]
        
        # Binary features (should be 0 or 1)
        self.binary_features = [
            'TLD_known', 'HasObfuscation', 'IsHTTPS', 'HasTitle', 'HasFavicon', 
            'Robots', 'IsResponsive', 'HasDescription', 'HasSocialNet', 
            'HasSubmitButton', 'HasHiddenFields', 'HasPasswordField', 
            'Bank', 'Pay', 'Crypto', 'HasCopyrightInfo', 'url_has_ip'
        ]
        
        # Continuous features that should be scaled
        self.continuous_features = [
            'URLLength', 'DomainLength', 'URLSimilarityIndex', 
            'CharContinuationRate', 'TLDLegitimateProb', 'URLCharProb', 
            'NoOfSubDomain', 'ObfuscationRatio', 'LetterRatioInURL', 
            'SpacialCharRatioInURL', 'DomainTitleMatchScore', 'URLTitleMatchScore', 
            'NoOfImage', 'NoOfCSS', 'NoOfJS', 'NoOfSelfRef', 'NoOfExternalRef',
            'domain_entropy'
        ]
    
    def load_and_validate_data(self, data_path):
        """Load and perform initial validation of the dataset."""
        print("Loading dataset...")
        
        if isinstance(data_path, str):
            # Load from file
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
                df = pd.read_excel(data_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
        else:
            # Assume it's already a DataFrame
            df = data_path.copy()
        
        print(f"Dataset shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Basic validation
        print(f"Duplicate rows: {df.duplicated().sum()}")
        print(f"Completely empty rows: {df.isnull().all(axis=1).sum()}")
        print(f"Completely empty columns: {df.isnull().all(axis=0).sum()}")
        
        return df
    
    def analyze_missing_data(self, df):
        """Analyze missing data patterns."""
        print("\n=== Missing Data Analysis ===")
        
        missing_stats = df.isnull().sum()
        missing_percent = (missing_stats / len(df)) * 100
        
        missing_summary = pd.DataFrame({
            'Missing_Count': missing_stats,
            'Missing_Percentage': missing_percent
        }).sort_values('Missing_Percentage', ascending=False)
        
        print("Features with missing values:")
        print(missing_summary[missing_summary['Missing_Count'] > 0])
        
        # Identify patterns
        url_missing = sum(1 for feat in self.url_features if feat in df.columns and df[feat].isnull().sum() > 0)
        content_missing = sum(1 for feat in self.content_features if feat in df.columns and df[feat].isnull().sum() > 0)
        
        print(f"\nMissing patterns:")
        print(f"URL features with missing values: {url_missing}")
        print(f"Content features with missing values: {content_missing}")
        
        return missing_summary
    
    def clean_initial_data(self, df):
        """Remove completely empty rows and columns."""
        print("\n=== Initial Data Cleaning ===")
        
        initial_shape = df.shape
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        print(f"Shape after cleaning: {df.shape} (removed {initial_shape[0] - df.shape[0]} rows, {initial_shape[1] - df.shape[1]} columns)")
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values based on feature types."""
        print("\n=== Handling Missing Values ===")
        
        df_processed = df.copy()
        
        # Handle binary features - fill with 0 (most conservative approach)
        for feature in self.binary_features:
            if feature in df_processed.columns:
                missing_count = df_processed[feature].isnull().sum()
                if missing_count > 0:
                    df_processed[feature] = df_processed[feature].fillna(0)
                    print(f"Filled {missing_count} missing values in {feature} with 0")
        
        # Handle continuous features - fill with median
        for feature in self.continuous_features:
            if feature in df_processed.columns:
                missing_count = df_processed[feature].isnull().sum()
                if missing_count > 0:
                    median_val = df_processed[feature].median()
                    df_processed[feature] = df_processed[feature].fillna(median_val)
                    self.feature_stats[feature] = {'median': median_val}
                    print(f"Filled {missing_count} missing values in {feature} with median: {median_val:.3f}")
        
        # Handle count features (NoOf*) - fill with 0
        count_features = [col for col in df_processed.columns if col.startswith('NoOf')]
        for feature in count_features:
            missing_count = df_processed[feature].isnull().sum()
            if missing_count > 0:
                df_processed[feature] = df_processed[feature].fillna(0)
                print(f"Filled {missing_count} missing values in {feature} with 0")
        
        # Handle ratio/score features - fill with median or appropriate default
        ratio_features = [col for col in df_processed.columns if 'Ratio' in col or 'Score' in col or 'Prob' in col]
        for feature in ratio_features:
            if feature in df_processed.columns:
                missing_count = df_processed[feature].isnull().sum()
                if missing_count > 0:
                    if feature in self.binary_features:
                        fill_value = 0
                    else:
                        fill_value = df_processed[feature].median()
                    df_processed[feature] = df_processed[feature].fillna(fill_value)
                    print(f"Filled {missing_count} missing values in {feature} with {fill_value:.3f}")
        
        return df_processed
    
    def validate_data_types(self, df):
        """Validate and correct data types."""
        print("\n=== Validating Data Types ===")
        
        df_processed = df.copy()
        
        # Ensure binary features are 0 or 1
        for feature in self.binary_features:
            if feature in df_processed.columns:
                # Convert to numeric first
                df_processed[feature] = pd.to_numeric(df_processed[feature], errors='coerce')
                # Clip to 0-1 range and round
                df_processed[feature] = df_processed[feature].clip(0, 1).round()
                
                # Check for invalid values
                invalid_count = df_processed[feature].isnull().sum()
                if invalid_count > 0:
                    print(f"Warning: {invalid_count} invalid values in binary feature {feature}, filling with 0")
                    df_processed[feature] = df_processed[feature].fillna(0)
        
        # Ensure continuous features are numeric
        for feature in self.continuous_features:
            if feature in df_processed.columns:
                df_processed[feature] = pd.to_numeric(df_processed[feature], errors='coerce')
                
                # Handle any remaining NaN values
                if df_processed[feature].isnull().sum() > 0:
                    median_val = df_processed[feature].median()
                    df_processed[feature] = df_processed[feature].fillna(median_val)
        
        return df_processed
    
    def detect_and_handle_outliers(self, df, method='iqr', threshold=1.5):
        """Detect and handle outliers in continuous features."""
        print(f"\n=== Outlier Detection and Handling (method: {method}) ===")
        
        df_processed = df.copy()
        outlier_summary = {}
        
        for feature in self.continuous_features:
            if feature in df_processed.columns:
                Q1 = df_processed[feature].quantile(0.25)
                Q3 = df_processed[feature].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((df_processed[feature] < lower_bound) | 
                           (df_processed[feature] > upper_bound))
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    # Cap outliers instead of removing them
                    df_processed[feature] = df_processed[feature].clip(lower_bound, upper_bound)
                    outlier_summary[feature] = {
                        'count': outlier_count,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                    print(f"Capped {outlier_count} outliers in {feature} to [{lower_bound:.3f}, {upper_bound:.3f}]")
        
        return df_processed, outlier_summary
    
    def scale_features(self, df, method='standard'):
        """Scale continuous features."""
        print(f"\n=== Feature Scaling (method: {method}) ===")
        
        df_processed = df.copy()
        
        # Only scale continuous features
        features_to_scale = [f for f in self.continuous_features if f in df_processed.columns]
        
        if len(features_to_scale) == 0:
            print("No continuous features to scale")
            return df_processed
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Scaling method must be 'standard' or 'minmax'")
        
        # Fit and transform
        scaled_features = self.scaler.fit_transform(df_processed[features_to_scale])
        df_processed[features_to_scale] = scaled_features
        
        print(f"Scaled {len(features_to_scale)} continuous features using {method} scaling")
        
        return df_processed
    
    def final_validation(self, df):
        """Perform final validation of the processed dataset."""
        print("\n=== Final Dataset Validation ===")
        
        # Check for any remaining missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values still present")
            print(df.isnull().sum()[df.isnull().sum() > 0])
        else:
            print("✓ No missing values")
        
        # Check data types
        print(f"✓ Dataset shape: {df.shape}")
        print(f"✓ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check binary features
        for feature in self.binary_features:
            if feature in df.columns:
                unique_vals = set(df[feature].unique())
                if not unique_vals.issubset({0, 1}):
                    print(f"Warning: {feature} has non-binary values: {unique_vals}")
        
        # Check for infinite values
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            print(f"Warning: {inf_count} infinite values present")
        else:
            print("✓ No infinite values")
        
        # Summary statistics
        print("\nDataset summary:")
        print(f"Total features: {df.shape[1]}")
        print(f"Binary features: {len([f for f in self.binary_features if f in df.columns])}")
        print(f"Continuous features: {len([f for f in self.continuous_features if f in df.columns])}")
        
        return df
    
    def preprocess_dataset(self, data_path, handle_outliers=True, scale_features=True, scaling_method='standard'):
        """
        Complete preprocessing pipeline.
        
        Parameters:
        -----------
        data_path : str or DataFrame
            Path to the dataset file or DataFrame
        handle_outliers : bool
            Whether to handle outliers
        scale_features : bool
            Whether to scale continuous features
        scaling_method : str
            Scaling method ('standard' or 'minmax')
        
        Returns:
        --------
        tuple: (preprocessed_df, preprocessor_stats)
        """
        
        print("Starting comprehensive preprocessing pipeline...")
        print("=" * 60)
        
        # Load and validate
        df = self.load_and_validate_data(data_path)
        
        # Analyze missing data
        missing_summary = self.analyze_missing_data(df)
        
        # Initial cleaning
        df = self.clean_initial_data(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Validate data types
        df = self.validate_data_types(df)
        
        # Handle outliers
        outlier_summary = {}
        if handle_outliers:
            df, outlier_summary = self.detect_and_handle_outliers(df)
        
        # Scale features
        if scale_features:
            df = self.scale_features(df, method=scaling_method)
        
        # Final validation
        df = self.final_validation(df)
        
        # Compile preprocessing statistics
        preprocessor_stats = {
            'missing_summary': missing_summary,
            'outlier_summary': outlier_summary,
            'feature_stats': self.feature_stats,
            'scaler': self.scaler,
            'feature_categories': {
                'binary': self.binary_features,
                'continuous': self.continuous_features,
                'url': self.url_features,
                'content': self.content_features,
                'security': self.security_features
            },
            'final_shape': df.shape
        }
        
        print("\n" + "=" * 60)
        print("Preprocessing completed successfully!")
        print(f"Final dataset shape: {df.shape}")
        
        return df, preprocessor_stats
    
    def save_preprocessed_data(self, df, output_path='preprocessed_dataset.csv'):
        """Save preprocessed data to file."""
        df.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to {output_path}")
        
    def transform_new_data(self, new_data):
        """
        Apply the same preprocessing steps to new data using fitted parameters.
        
        Parameters:
        -----------
        new_data : DataFrame
            New data to transform
            
        Returns:
        --------
        DataFrame: Transformed data
        """
        if self.scaler is None:
            raise ValueError("Preprocessor must be fitted first. Call preprocess_dataset() first.")
        
        df = new_data.copy()
        
        # Handle missing values using the same logic
        df = self.handle_missing_values(df)
        df = self.validate_data_types(df)
        
        # Apply scaling using fitted scaler
        features_to_scale = [f for f in self.continuous_features if f in df.columns]
        if len(features_to_scale) > 0 and self.scaler is not None:
            df[features_to_scale] = self.scaler.transform(df[features_to_scale])
        
        return df


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = PhishingDataPreprocessor()
    
    # Example: Load and preprocess data
    # Replace 'your_dataset.csv' with your actual file path
    try:
        # Preprocess the dataset
        preprocessed_df, stats = preprocessor.preprocess_dataset(
            data_path='outputs/datasets/final_combined_dataset.csv',  # Replace with your file path
            handle_outliers=True,
            scale_features=True,
            scaling_method='standard'
        )
        
        # Save preprocessed data
        preprocessor.save_preprocessed_data(preprocessed_df, 'preprocessed_phishing_dataset.csv')
        
        print(f"\nPreprocessing completed successfully!")
        print(f"Preprocessed dataset shape: {preprocessed_df.shape}")
        print(f"Dataset saved as: preprocessed_phishing_dataset.csv")
        
        # Display basic info about the preprocessed dataset
        print(f"\nFeature summary:")
        print(f"Total features: {preprocessed_df.shape[1]}")
        print(f"Total samples: {preprocessed_df.shape[0]}")
        
        # Show sample of preprocessed data
        print(f"\nFirst few rows of preprocessed data:")
        print(preprocessed_df.head())
        
    except FileNotFoundError:
        print("Please replace 'your_dataset.csv' with the actual path to your dataset")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("Please check your dataset format and file path")