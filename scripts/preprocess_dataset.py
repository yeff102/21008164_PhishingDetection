import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Set the path to the dataset
csv_path = './data/phishing_subset_25000.csv'

try:
    df = pd.read_csv(csv_path)

    # Basic summary
    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn Info:")
    print(df.info())

    # Step 1: Inspect the structure and class distribution
    print("\nColumn names:", df.columns.tolist())

    if 'label' in df.columns:
        print("\nClass distribution:")
        print(df['label'].value_counts())
    else:
        print("\nNo 'label' column found. Please check the column name for target values.")

    # Step 2a: Clean noise & duplicates
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    removed = initial_rows - df.shape[0]
    print(f"\nDuplicates removed: {removed}")

    # Step 2b: Handle missing values
    print("\nMissing values per column (before):")
    print(df.isnull().sum())

    # Drop rows with too many missing values or fill with defaults
    df.dropna(thresh=int(0.7 * df.shape[1]), inplace=True)  # drop if more than 30% missing
    df.fillna(method='ffill', inplace=True)  # forward fill as default strategy

    print("\nMissing values per column (after):")
    print(df.isnull().sum())

    # Step 2c: Normalize and encode data
    if 'URL' in df.columns or 'url' in df.columns:
        url_column = 'URL' if 'URL' in df.columns else 'url'

        # Encode: Has HTTPS
        df['Has_HTTPS'] = df[url_column].str.startswith('https').astype(int)

        # Feature: URL Length
        df['URL_Length'] = df[url_column].apply(len)

        # Normalize: URL_Length
        scaler = MinMaxScaler()
        df[['URL_Length']] = scaler.fit_transform(df[['URL_Length']])

        print("\nSample extracted and normalized features:")
        print(df[['URL_Length', 'Has_HTTPS']].head())
    else:
        print("\nURL column not found. Please check the column name.")

    # Save the cleaned and preprocessed version
    df.to_csv('./outputs/preprocessed_dataset.csv', index=False)
    print("\nPreprocessed dataset saved to './outputs/preprocessed_dataset.csv'.")

except FileNotFoundError:
    print("File not found. Please check the path.")
except Exception as e:
    print("Error loading file:", e)