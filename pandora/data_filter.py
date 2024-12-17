import pandas as pd

# Define valid MBTI types
VALID_MBTIs = {'INTJ', 'INTP', 'INFP', 'ENTP', 'ISTP', 'ISFP', 'ESTJ', 'ISTJ', 'ESTP', 'ISFJ', 'ENFP', 'ESFP', 'ESFJ', 'ENFJ', 'INFJ', 'ENTJ'}

# Define input file path
input_path = "./pandora_comments/processed_comments.csv"

# Read data
data = pd.read_csv(input_path)

# Filter out invalid MBTI types
filtered_data = data[data['type'].isin(VALID_MBTIs)]

# To check for excluded data, use the following
invalid_data = data[~data['type'].isin(VALID_MBTIs)]
if not invalid_data.empty:
    print("The following data contains invalid MBTI types and has been excluded:")
    print(invalid_data)

# Define output file path
output_path = "./pandora_comments/filtered_processed_comments.csv"

# Write results to CSV file
filtered_data.to_csv(output_path, index=False)

print("Data has been validated and processed, saved to:", output_path)
