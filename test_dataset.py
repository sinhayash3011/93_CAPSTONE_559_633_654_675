from dataset_loader import load_dataset, split_dataset

# Load dataset
df = load_dataset("IMU.csv")
print("Columns:", df.columns)
print("First 5 rows:\n", df.head())

# Split dataset
X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.2)

print("Train Features shape:", X_train.shape)
print("Test Features shape:", X_test.shape)
print("Train Labels shape:", y_train.shape)
print("Test Labels shape:", y_test.shape)
