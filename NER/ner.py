'''
 The code partially uses Sophie Francis' code https://github.com/firojalam/crisis_datasets_benchmarks/tree/master/bin
'''

import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Input and output JSON file paths
input_json_file = "../classifier/output_twitter.json"
output_json_file = "output_twitter.json"

# Specify the model path
model_path = './models/roberta-base-new-bio-updated-ner-5.model/'

# Categories to be retained (disaster-related)
relevant_categories = [
    "injured_or_dead_people",
    "infrastructure_and_utilities_damage",
    "affected_individual",
    "displaced_and_evacuations",
    "missing_and_found_people"
]

# NER Tag List
label_list = ['O', 'B-IMPACT', 'I-IMPACT', 'B-AFFECTED', 'I-AFFECTED', 'B-SEVERITY', 'I-SEVERITY', 'B-LOCATION',
              'I-LOCATION', 'B-MODIFIER', 'I-MODIFIER']

# Load JSON data
def load_data(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

# Filter the text of tweets containing relevant categories
def filter_tweets(data, relevant_categories):
    return [item["text"] for item in data if item["predicted_label"] in relevant_categories]

# Processing individual tweets
def process_tweet(sentence, tokenizer, model, label_list):
    # Tokenize the sentence
    tokens = tokenizer(sentence)
    torch.tensor(tokens['input_ids']).unsqueeze(0).size()

    # Model reasoning
    preds = model.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0),
                          attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
    preds = torch.argmax(preds.logits.squeeze(), axis=1)

    impacts = tokenizer.batch_decode(tokens['input_ids'])
    value_preds = [label_list[i] for i in preds]

    # Recombine the split tokens into sentences and compute the start and end indices of each word.
    sentence_spite ={"word": "", "label": "","start":0,"end":0}
    current_index = 0
    entities_sentence=[]
    sentence_token=""
    for impact, label in zip(impacts, value_preds):
        sentence_token+=impact
        len_impact=len(impact)
        if impact.startswith(" "):
            current_index+=1
            len_impact-=1

        sentence_spite = {
            "word": impact,
            "label": label,
            "start": current_index,
            "end": current_index + len_impact
        }
        current_index += len_impact
        entities_sentence.append(sentence_spite)

    return entities_sentence, sentence_token

# # Processing entity information
def process_entities(entities_sentence):
    last_label ="O"
    current_entity = {"impact": "", "label": "","start":0,"end":0}
    entities = []

    for entity in entities_sentence:
        label=entity["label"]
        word=entity["word"]
        if label == "O":
            last_label = label
            # Skip "O" labels
            continue
        label = label.split("-", 1)[-1]  # Remove content before "-"
        if label == last_label :
            # Append word to the current entity
            current_entity["impact"] += word
            current_entity["end"]+=len(word)
        else:
            # Start a new entity
            if current_entity["impact"]:
                entities.append(current_entity)
            current_entity = {"impact": word, "label": label,"start":entity["start"],"end":entity["end"]}
        last_label=label

    # Append the last entity
    if current_entity["impact"]:
        entities.append(current_entity)

    return entities

# Convert results to JSON format
def convert_to_json(results):
    converted_data = []
    for item in results:
        text_raw = item["text_raw"]
        text = item["text"]
        entities = item["entities"]

        converted_entities = []
        for entity in entities:
            word = entity["impact"]
            label = entity["label"]
            start_index = entity["start"]
            end_index = entity["end"]
            entity_info = [start_index, end_index, word, label]
            converted_entities.append(entity_info)

        converted_item = {
            "text_raw": text_raw,
            "text": text,
            "entities": converted_entities
        }

        converted_data.append(converted_item)

    return json.dumps(converted_data, ensure_ascii=False, indent=2)


# Initialise the model and splitter
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(label_list))

# Load data
data = load_data(input_json_file)
# Filter tweets containing relevant categories
filtered_tweets = filter_tweets(data, relevant_categories)

results = []
# Process each filtered tweet
for tweet in filtered_tweets:
    entities, sentence_token = process_tweet(tweet, tokenizer, model, label_list)
    processed_entities = process_entities(entities)

    # Build the output
    sentence_result = {
        "text_raw": tweet,
        "text": sentence_token, # This text is the reorganised text after word splitting
        "entities": processed_entities
    }

    results.append(sentence_result)

# Convert the result to JSON format and write it to a file
output_json = convert_to_json(results)
with open(output_json_file, 'w', encoding='utf-8') as json_file:
    json_file.write(output_json)
