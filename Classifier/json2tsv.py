import json
import csv
import os

# Specify the paths to the input and output files
input_folder = './retrieved_data/'
output_tsv_file = 'twitter_data_de.tsv'

# Create a list to store the data of all JSON files
all_data = []

# List all JSON files in the input folder
json_files = [file for file in os.listdir(input_folder) if file.endswith('.json')]

# Iterate through each JSON file and merge its data into the all_data list
for json_file in json_files:
    file_path = os.path.join(input_folder, json_file)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            item['text'] = item['text'].replace('\n', '<<newline>>')
        all_data.extend(data)

# Create a collection to track occurrences of text
seen_texts = set()

# Write all unduplicated data to tsv file
with open(output_tsv_file, 'w', encoding='utf-8', newline='') as tsv_file:
    writer = csv.writer(tsv_file, delimiter='\t')

    # Write in the title line
    writer.writerow(['id', 'created_at', 'text', 'author_id'])

    # Iterate through the merged data and write to TSV file
    for item in all_data:
        text = item['text']

        # If text is not in the collection, write it to the TSV file and add it to the collection
        if text not in seen_texts:
            seen_texts.add(text)

            id = item['id']
            created_at = item['created_at']
            author_id = item['author_id']

            # Write data to TSV file
            writer.writerow([id, created_at, text, author_id])

print(f"Conversion complete. Merged and deduplicated data saved to {output_tsv_file}")

