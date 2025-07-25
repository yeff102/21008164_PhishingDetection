import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_split_dataset(file_path, test_size=0.15, val_size=0.15, random_state=42):
    """
    Load and split the phishing detection dataset into train, validation, and test sets.
    
    Parameters:
    file_path (str): Path to the CSV file
    test_size (float): Proportion of dataset for test set (default: 0.15)
    val_size (float): Proportion of dataset for validation set (default: 0.15)  
    random_state (int): Random seed for reproducibility
    
    Returns:
    tuple: (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
    """
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # Display basic information about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Data types:")
    print(df.dtypes.value_counts())
    print(f"\nTarget distribution:")
    print(df['label'].value_counts())
    print(f"Percentage of phishing sites: {df['label'].mean():.2%}")
    
    # Check for non-numeric columns
    non_numeric = df.select_dtypes(include=['object']).columns.tolist()
    if 'label' in non_numeric:
        non_numeric.remove('label')
    if non_numeric:
        print(f"\nNon-numeric feature columns: {non_numeric}")
        for col in non_numeric:
            print(f"  {col}: {df[col].nunique()} unique values - {df[col].unique()[:10]}")  # Show first 10 unique values
    
    # One-Hot Encode the 'TLD' column if present
    if 'TLD' in df.columns:
        print("\nApplying One-Hot Encoding to 'TLD' column...")
        df = pd.get_dummies(df, columns=['TLD'], prefix='TLD', drop_first=False)
        print(f"'TLD' column successfully one-hot encoded. New shape: {df.shape}")
    
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Get feature names for later use
    feature_names = X.columns.tolist()
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # Second split: separate train and validation sets
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_size_adjusted, 
        random_state=random_state, 
        stratify=y_temp
    )
    
    # Print split information
    print(f"\nDataset split summary:")
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df):.1%})")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(df):.1%})")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df):.1%})")
    
    # Check class distribution in each set
    print(f"\nClass distribution:")
    print(f"Train - Phishing: {y_train.mean():.3f}, Legitimate: {1-y_train.mean():.3f}")
    print(f"Val - Phishing: {y_val.mean():.3f}, Legitimate: {1-y_val.mean():.3f}")
    print(f"Test - Phishing: {y_test.mean():.3f}, Legitimate: {1-y_test.mean():.3f}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

def scale_features(X_train, X_val, X_test):
    """
    Scale features using StandardScaler fitted on training data.
    Only scales numerical columns, leaves categorical columns unchanged.
    
    Parameters:
    X_train, X_val, X_test: Feature matrices
    
    Returns:
    tuple: (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    # Identify numerical and categorical columns
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols.tolist()}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols.tolist()}")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Create copies to avoid modifying original data
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    # Scale only numerical columns
    if len(numerical_cols) > 0:
        X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_val_scaled[numerical_cols] = scaler.transform(X_val[numerical_cols])
        X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        print(f"Scaled {len(numerical_cols)} numerical features using StandardScaler")
    else:
        print("No numerical features found to scale")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def visualize_data_distribution(y_train, y_val, y_test):
    """
    Visualize the class distribution across train, validation, and test sets.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = [y_train, y_val, y_test]
    titles = ['Training Set', 'Validation Set', 'Test Set']
    
    for i, (data, title) in enumerate(zip(datasets, titles)):
        counts = data.value_counts()
        axes[i].pie(counts.values, labels=['Legitimate', 'Phishing'], autopct='%1.1f%%')
        axes[i].set_title(f'{title}\n(n={len(data)})')
    
    plt.tight_layout()
    plt.show()

def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, output_dir='data_splits'):
    """
    Save the split datasets to separate NPY files.
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training set
    np.save(f'{output_dir}/X_train.npy', X_train.values)
    np.save(f'{output_dir}/y_train.npy', y_train.values)
    
    # Save validation set
    np.save(f'{output_dir}/X_val.npy', X_val.values)
    np.save(f'{output_dir}/y_val.npy', y_val.values)
    
    # Save test set
    np.save(f'{output_dir}/X_test.npy', X_test.values)
    np.save(f'{output_dir}/y_test.npy', y_test.values)
    
    print(f"Datasets saved as NPY files to {output_dir}/ directory")

# Main execution
if __name__ == "__main__":
    # Set file path to your dataset
    file_path = r"C:\Users\jasud\OneDrive - Sunway Education Group\school\sem 8\CP2\Phishing detection\outputs\datasets\final_preprocessed_dataset.csv"
    
    # Split the dataset
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_split_dataset(
        file_path, 
        test_size=0.15,    # 15% for testing
        val_size=0.15,     # 15% for validation
        random_state=42
    )
    
    # Optional: Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )
    
    # Visualize data distribution
    visualize_data_distribution(y_train, y_val, y_test)
    
    # Optional: Save splits to NPY files
    save_splits(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("\nDataset splitting completed!")
    print(f"Available variables:")
    print(f"- X_train, X_val, X_test: Feature matrices")
    print(f"- y_train, y_val, y_test: Target vectors")
    print(f"- X_train_scaled, X_val_scaled, X_test_scaled: Scaled feature matrices")
    print(f"- scaler: Fitted StandardScaler object")
    print(f"- feature_names: List of feature names")
    print(f"\nTo load saved datasets:")
    print(f"X_train = np.load('data_splits/X_train.npy')")
    print(f"y_train = np.load('data_splits/y_train.npy')")
    print(f"# Similar for val and test sets")