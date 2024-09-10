import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV data
# file_path = '/home/users/panjia/LLM-IHS-Explanation-master/exp_data/malicious_prompt.csv'  # Replace with your actual CSV file path
file_path = '/home/users/panjia/LLM-IHS-Explanation-master/exp_data/normal_prompt.csv'  # Replace with your actual CSV file path
df = pd.read_csv(file_path)

# Split the data into training and testing sets (30% test, 70% train)
train, test = train_test_split(df, test_size=0.3, random_state=42)

# Save the split data to new CSV files
train.to_csv('/home/users/panjia/LLM-IHS-Explanation-master/exp_data/train_normal_prompt.csv', index=False)
test.to_csv('/home/users/panjia/LLM-IHS-Explanation-master/exp_data/test_normal_prompt.csv', index=False)

print("Data has been split and saved to 'train_set.csv' and 'test_set.csv'.")
