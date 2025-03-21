import pandas as pd
# Read the CSV files into DataFrames
df = pd.read_csv("dat1.csv")
df2 = pd.read_csv('NewData.csv')

# Merge the DataFrames on their index (you can change the merge method as needed)
merged_df_outer = pd.merge(df, df2, left_index=True, right_index=True, how='outer')

# Display the merged DataFrame before cleaning
print("Merged DataFrame before cleaning:")
print(merged_df_outer)

# Drop columns with 'Unnamed' in their name (e.g., 'Unnamed: 0_x', 'Unnamed: 0_y')
merged_df_outer.drop(columns=[col for col in merged_df_outer.columns if 'Unnamed' in col], inplace=True, errors='ignore')

# If you want to keep the '_x' and '_y' suffixed columns (like 'label_x', 'label_y', etc.), we leave them as is

# Optionally, drop other specific columns that are not needed
# Here I assume you want to drop columns like 'speech' and 'file_path' which are unnecessary
merged_df_outer.drop(columns=['speech'], inplace=True, errors='ignore')
merged_df_outer.drop(columns=['file_path'], inplace=True, errors='ignore')
merged_df_outer.drop(columns=['label_x'], inplace=True, errors='ignore')
# Reset the index if necessary (optional)
merged_df_outer.reset_index(drop=True, inplace=True)

# Save the cleaned DataFrame to a new CSV
merged_df_outer.to_csv("merged_cleaned.csv", index=False)

# Display the cleaned DataFrame
print("Cleaned DataFrame:")
print(merged_df_outer)
