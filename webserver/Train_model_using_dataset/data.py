import pandas as pd

df = pd.read_csv("dat1.csv")
df2 = pd.read_csv('NewData.csv')

merged_df_outer = pd.merge(df, df2, left_index=True, right_index=True, how='outer')


print("Merged DataFrame before cleaning:")
print(merged_df_outer)
merged_df_outer.drop(columns=[col for col in merged_df_outer.columns if 'Unnamed' in col], inplace=True, errors='ignore')

merged_df_outer.drop(columns=['speech'], inplace=True, errors='ignore')
merged_df_outer.drop(columns=['file_path'], inplace=True, errors='ignore')
merged_df_outer.drop(columns=['label_x'], inplace=True, errors='ignore')

merged_df_outer.reset_index(drop=True, inplace=True)

# Save the cleaned DataFrame to a new CSV
merged_df_outer.to_csv("merged_cleaned.csv", index=False)

