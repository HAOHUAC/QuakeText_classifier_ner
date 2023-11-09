'''
 The code partially uses Firoj Alam's code https://github.com/firojalam/crisis_datasets_benchmarks/tree/master/bin
'''

import torch
import csv
import os
import json
from tqdm import tqdm
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features_multiclass as convert_examples_to_features
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertForSequenceMultiClassClassification, BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          DistilBertConfig,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer)

# Define file paths and parameters of the model
args = {
    "test_file":"twitter_data_de.tsv",
    "output_json_file":"output_twitter.json",
    "model_name_or_path":"output/checkpoint-10150_roberta",
    "token":"roberta_token",
    "local_rank":-1,
    "max_seq_length": 128,
    "model_type" : "roberta",
    "test_batch_size": 8,
    "data_dir": "data_roberta_model_hum",
    "overwrite_cache":True,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# Define all available models
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceMultiClassClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}

# Define the numerical mapping of categories
label_mapping = {
    0: "affected_individual",
    1: "caution_and_advice",
    2: "displaced_and_evacuations",
    3: "donation_and_volunteering",
    4: "infrastructure_and_utilities_damage",
    5: "injured_or_dead_people",
    6: "missing_and_found_people",
    7: "not_humanitarian",
    8: "requests_or_needs",
    9: "response_efforts",
    10: "sympathy_and_support",
}

# Use GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_cache_examples(args, task, tokenizer, evaluate=False):

    if args['local_rank'] not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    # Selection of the appropriate processor for the task
    processor = processors[task]()

    output_mode = output_modes[task]  # Get output mode

    # Extract the base name from the test file path (without path and extension)
    base_name = os.path.basename(args['test_file'])
    base_name = os.path.splitext(base_name)[0]
    dataset = base_name

    # Build the cache file path
    cached_features_file = os.path.join(args['data_dir'], 'cached_{}_{}_{}_{}'.format(dataset,
        list(filter(None, args['model_name_or_path'].split('/'))).pop(),
        str(args['max_seq_length']),
        str(task)))

    # If the cache file exists and does not overwrite the cache, load the cached features
    if os.path.exists(cached_features_file) and not args['overwrite_cache']:
        features = torch.load(cached_features_file)
    else:

        # Get test examples from the processor
        examples = processor.get_test_examples(args['test_file'])

        # Get a list of tags
        label_list = processor.get_labels()

        # Fixes tag indexes being swapped in RoBERTa pre-trained models
        if task in ['mnli', 'mnli-mm'] and args['model_type'] in ['roberta']:
            label_list[1], label_list[2] = label_list[2], label_list[1]

        # Convert examples to features
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args['max_seq_length'],
                                                output_mode=output_mode,
                                                pad_on_left=bool(args['model_type'] in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0,
        )
        # If first process, save feature to cache file
        if args['local_rank'] in [-1, 0]:
            torch.save(features, cached_features_file)
    # If first process in distributed training and not evaluating, wait for first process to process dataset while others use cache
    if args['local_rank'] == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    # Select label tensor type based on output mode
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return dataset

# Open a TSV file and read the text columns
tweets = []
with open(args["test_file"], 'r', encoding='utf-8') as tsv_file:
    reader = csv.DictReader(tsv_file, delimiter='\t')
    for row in reader:
        tweets.append(row['text'])

# Specify the model and tokenizer to be used
config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
model = model_class.from_pretrained(args['model_name_or_path'])
tokenizer = tokenizer_class.from_pretrained(args['token'], do_lower_case=True)

# Call load_and_cache_examples function to load data
dataset = load_and_cache_examples(args, "multiclass", tokenizer, evaluate=True)
test_sampler = SequentialSampler(dataset)
test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=args['test_batch_size'])

results = []  # Initialize a list to store results
input_ids_list = []  # Initialize the list to store input_ids
attention_masks_list = []  # Initialise the list used to store attention_masks
max_seq_length = args['max_seq_length']

# Initialize a counter for tracking batches
j = 0

# Loop over batches in the test dataloader
for batch in tqdm(test_dataloader, desc="Classifying"):
    # Move batch elements to the specified device (CPU or GPU)
    batch = tuple(t.to(args['device']) for t in batch)

    # Prepare inputs for the model
    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1]}

    # Disable gradient calculations during inference
    with torch.no_grad():
        outputs = model(**inputs)  # Forward pass through the model

    # Extract logits from the model outputs
    logits = outputs[0]
    # Apply softmax to obtain class probabilities
    class_probabilities = torch.softmax(logits, dim=1)
    # Convert numeric labels to a list of predicted labels
    predicted_labels_numeric= class_probabilities.argmax(dim=1).tolist()
    # Map numeric labels to human-readable labels using label_mapping
    predicted_labels = [label_mapping.get(label, "UnknownLabel") for label in predicted_labels_numeric]

    # Iterate through each example in the batch
    for i, predicted_label in enumerate(predicted_labels):
        text = tweets[i+j]
        # create a result dictionary containing the text and predicted labels
        result = {"text": text, "predicted_label": predicted_label}
        results.append(result)

    # Increment the counter by the batch size
    j += args['test_batch_size']

# Write the classification result as a json file
with open(args["output_json_file"], "w") as json_file:
    json.dump(results, json_file, indent=4)

