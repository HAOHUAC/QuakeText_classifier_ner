import pandas as pd
import json

# Specify input and output file paths
input_tsv_file = '../classifier/twitter_data_de.tsv'
input_json_file = 'output_twitter.json'
output_json_file = 'merge_twitter.json'

# Reading data from a TSV file
tsv_data = pd.read_csv(input_tsv_file, delimiter='\t')

# Reading data from a JSON file
with open(input_json_file, 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)
json_df = pd.DataFrame(json_data)

# Use the merge function to combine two data sources based on text columns
merged_data = pd.merge(tsv_data, json_df, left_on='text', right_on='text_raw', how='right')

# Select the columns to keep
merged_data = merged_data[['id','created_at','text_y', 'entities']]

# Renaming columns
merged_data.columns = ['id', 'created_at','text', 'entities']
merged_data = merged_data.to_dict(orient='records')
output_json = json.dumps(merged_data, ensure_ascii=False, indent=2)

# Save the JSON to a file
with open(output_json_file, 'w', encoding='utf-8') as json_file:
    json_file.write(output_json)

