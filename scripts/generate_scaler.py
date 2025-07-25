import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Step 1: Load training data
X_train = np.load(r"C:\Users\jasud\OneDrive - Sunway Education Group\school\sem 8\CP2\Phishing detection\data_splits\X_train.npy")

# Step 2: Fit the scaler
scaler = StandardScaler()
scaler.fit(X_train)

# Step 3: Save the scaler to the model directory
output_path = r"C:\Users\jasud\OneDrive - Sunway Education Group\school\sem 8\CP2\Phishing detection\models\training_20250718_160202\scaler.pkl"
with open(output_path, "wb") as f:
    pickle.dump(scaler, f)

print("✓ Scaler successfully created and saved to:", output_path)
