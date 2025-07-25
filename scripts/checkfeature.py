import pandas as pd
# Load the same DataFrame you used before creating X_train
df = pd.read_csv("outputs/datasets/final_preprocessed_dataset.csv")  # Or however you loaded data

X = df.drop(columns=["label"])  # or however you sliced it
print("feature_31 is:", X.columns[31])
