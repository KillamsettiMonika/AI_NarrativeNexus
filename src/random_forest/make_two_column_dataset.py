import pandas as pd
import os

# Path to your original dataset
input_path = "cleaned_dataset.csv"   # adjust if different

# Path to save the new dataset inside random_forest folder
output_path = os.path.join("src", "random_forest", "two_column_dataset.csv")

def make_two_column_dataset():
    # Load your dataset
    df = pd.read_csv(input_path)

    # Adjust based on your dataset structure:
    # Suppose you have a column "text" and "category"
    # If not, replace with the correct column names.
    if "text" in df.columns and "category" in df.columns:
        new_df = df[["text", "category"]]
    else:
        # If dataset looks different, rename accordingly
        # Example: assume first column = label, second column = text
        new_df = df.iloc[:, [0, 1]]
        new_df.columns = ["text", "category"]

    # Save the cleaned two-column dataset
    new_df.to_csv(output_path, index=False)
    print(f"✅ Two-column dataset saved at: {output_path}")

if __name__ == "__main__":
    make_two_column_dataset()
