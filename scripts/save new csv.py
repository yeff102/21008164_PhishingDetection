import pandas as pd

# Load the full dataset
df = pd.read_csv('./data/PhiUSIIL_Phishing_URL_Dataset.csv')

# Select the first 25,000 rows
df_subset = df.head(25000)

# Save to a new CSV file
df_subset.to_csv('./data/phishing_subset_25000.csv', index=False)

print("Saved first 25,000 rows to phishing_subset_25000.csv")
