import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\jasud\OneDrive - Sunway Education Group\school\sem 8\CP2\Phishing detection\outputs\datasets\feature_extracted_dataset.csv")

# 1. Class balance
print("\n📊 Class Distribution:")
print(df['label'].value_counts())

# 2. Missing values
print("\n🚨 Top Features with Missing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# 3. Low variance features
low_var = df.loc[:, df.nunique() <= 1]
print(f"\n🧊 Low-variance features ({len(low_var.columns)}):")
print(list(low_var.columns))

# 4. Correlation heatmap
print("\n📈 Showing correlation heatmap...")
numeric_df = df.select_dtypes(include=['number'])  # filter only numeric columns
corr = numeric_df.corr().abs()
sns.heatmap(corr, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()
