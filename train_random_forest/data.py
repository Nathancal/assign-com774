import pandas as pd

# Replace 'file1.csv', 'file2.csv', 'file3.csv' with your actual file names
file1 = pd.read_csv('Sub2_Ajay.csv')
file2 = pd.read_csv('Sub3_Bhargav.csv')
file3 = pd.read_csv('Sub4_Hussain.csv')

# Concatenate the dataframes row-wise
combined_df = pd.concat([file1, file2, file3], axis=0, ignore_index=True)

# Write the combined dataframe to a new CSV file
combined_df.to_csv('subject2-4.csv', index=False)
