import json

# Specify input and output paths
input_json_file = 'output_twitter.json'
output_json_file = 'twitter_loction.json'

# Loading JSON file data
with open(input_json_file, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Filter out eligible items
filtered_items = []
for item in data:
    has_location = False
    for entity in item["entities"]:
        # Exclusion of some place names
        if entity[3] == "LOCATION" and entity[2] not in [" Afghanistan", "Afghanistan","Morocco"," Morocco","Libya"," Libya"]:
            has_location = True
            break
    if has_location:
        filtered_items.append(item)

# Calculate the number of items in the original file
total_items = len(data)

# Calculate the number of filtered items
filtered_items_count = len(filtered_items)
print("Number of texts containing place names:", filtered_items_count)

# Write filter results to a new JSON file
with open(output_json_file, 'w', encoding='utf-8') as output_file:
    json.dump(filtered_items, output_file, indent=4)