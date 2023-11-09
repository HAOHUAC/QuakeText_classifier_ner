import json

# Define paths to input and output files
input_json_file = 'output_twitter.json'
output_txt_flie = "analyze_output_twitter.txt"

# Define all categories
categories = [
    "affected_individual",
    "caution_and_advice",
    "displaced_and_evacuations",
    "donation_and_volunteering",
    "infrastructure_and_utilities_damage",
    "injured_or_dead_people",
    "missing_and_found_people",
    "not_humanitarian",
    "requests_or_needs",
    "response_efforts",
    "sympathy_and_support"
]

# Define disaster-related categories
relevant_categories = [
    "injured_or_dead_people",
    "infrastructure_and_utilities_damage",
    "affected_individual",
    "missing_and_found_people",
    "displaced_and_evacuations"
]

# Load data from JSON file
with open(input_json_file, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Create a dictionary for consolidating text by category
category_data = {category: [] for category in categories}

# Integration of data by category
for entry in data:
    label = entry["predicted_label"]
    text = entry["text"]
    if label in category_data:
        category_data[label].append(text)

# Output results to TXT file
with open(output_txt_flie, "w", encoding="utf-8") as output_file:
    total_num = 0
    relevant_counts = {label: 0 for label in categories}

    # Calculate the number of each category
    for label in categories:
        num_tweets = len(category_data[label])
        output_file.write(f"Category: {label}, Number of Tweets: {num_tweets}\n")
        total_num += num_tweets

    output_file.write("\n")
    output_file.write("Relevant categories:\n")

    # Calculate the number of relevant and irrelevant texts
    for label in relevant_categories:
        if label in relevant_counts:
            relevant_counts[label] = len(category_data[label])
            output_file.write(f"   {label}: {relevant_counts[label]}\n")

    relevant_num = sum(relevant_counts.values())
    irrelevant_num = total_num - relevant_num

    output_file.write("\n")
    output_file.write(f"Relevant tweets total: {relevant_num}\n")
    output_file.write(f"Irrelevant tweets: {irrelevant_num}\n")
    output_file.write(f"Total number of tweets: {total_num}\n")
    output_file.write("\n")

    #  Output text content for each category
    for label in categories:
        output_file.write(f"Category: {label}\n")
        output_file.write(f"Number of Tweets: {len(category_data[label])}\n")
        for text in category_data[label]:
            output_file.write(f"{text}\n")
        output_file.write("\n")
