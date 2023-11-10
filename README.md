# QuakeText_classifier_ner
### Haohua Chang
The model file pytorch_model.bin of the classifier and NER cannot be uploaded due to its large size. To run these models they need to be trained again on a local machine to regenerated this .bin file 

The classifier and the NER model will run in two different environments:

./Classifier and ./Classifier_model will be run in environment classifier_env.yaml

./NER and ./NER_model will be run in environment ner_env.yaml
## ./Classifier
These files perform data cleaning on the retrieved social media data and then use a classifier to classify them, and finally save the classification results as a json file.
### json2tsv.py
This code writes the contents of all the json files in the folder (retrieved_data) to a tsv file and removes all textual duplicates.
- The json file contains the obtained social media data.
- The tsv file is the format required by the classifier
### Classifier.py
The code will read the tsv file output by json2tsv.py and load the trained Roberta model to predict the labels of the input data
- The code will only read the text of the tsv file.
- The Classification results will be stored in a json file
- The output json file contains all text and its corresponding category labels.
### analyze.py
This code will read the json file output by Classifier.py to analyze the classification results of the classifier
- The code will output a txt file with the number of texts in each category, the number of data related to the disaster, the number of texts not related to the disaster, the total number of texts, and the text content of each category
## ./NER
These files will use the NER model to identify disaster impact and location information for disaster-related data
### ner.py
The code will first read the json file output by ./Classifier/classifier.py, and then the code will filter the text in the disaster-related categories from the json file. Finally, the code will load the trained Roberta model to identify the disaster impact and location information in these texts
- The code writes the text to a json file with the recognised entities
- Since the NER model outputs BIO tags, the code integrates them into full labels and full entity text
- Each entity will contain the entity's position in this text (start and end indexes), the entity text and the entity labels
### analyze_location.py
The code will count the amount of text in the output json file of ner.py that contains the place names.
- This code writes the data containing the place names to a json file
- This code excludes country names when filtering text that contains place names
### merge.py
The code will read the tsv file output by ./Classifier.py/json2tsv.py and the json file output by ner.py, and fuses the information in the tsv file, such as publish time and tweet ID, into the json file by matching the same text.  
- This code will output a json file that incorporates the complete information
## ./Classifier_model
These files are used to train the classifier model
### script.ipynb
The code will be run in jupyter notebook, the code contains the training parameters with the tasks performed (training, testing, evaluation)
- The code will call run_glue_multiclass.py
### run_glue_multiclass.py
Code derived from： Firoj Alam https://github.com/firojalam/crisis_datasets_benchmarks/tree/master/bin
- The code will train the Roberta model using the data in the folder data
- The model will be saved to the output_multi_class_roberta_large_hum folder
- This project uses the model in checkpoint-10150_roberta
## ./NER_model
These files are used to train the named entity recognition model
### roberta.py
Code derived from：Sophie Francis https://github.com/masseygeoinformaticscollaboratory/quaketext
- The code will use the data in the folder training-updated-bio to perform a 10-fold cross-validation
- All models will be saved in the folder models



