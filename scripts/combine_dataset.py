import pandas as pd

# Load both datasets
existing_df = pd.read_csv(r"C:\Users\jasud\OneDrive - Sunway Education Group\school\sem 8\CP2\Phishing detection\outputs\datasets\enhanced_dataset.csv")
new_df = pd.read_csv("processed_url_dataset.csv")

# Combine
combined_df = pd.concat([existing_df, new_df], ignore_index=True)

# Save
combined_df.to_csv(r"C:\Users\jasud\OneDrive - Sunway Education Group\school\sem 8\CP2\Phishing detection\outputs\datasets\final_combined_dataset.csv", index=False)

print(f"Combined dataset: {combined_df.shape}")
print(combined_df['label'].value_counts())