# -*- coding: utf-8 -*-

import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, Trainer, TrainingArguments
import torch
import json
import ast
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import TrainerCallback
from copy import deepcopy

import matplotlib.pyplot as plt

from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

"""#### ***Data Load***"""

def extract_values(output_str):
    output_dict = ast.literal_eval(output_str)
    seed_type = output_dict.get('seed_type', None)
    crop_name = output_dict.get('crop_name', None)
    pest_name = output_dict.get('pest_name',  None)
    return seed_type, crop_name, pest_name

data_files = {"train": "train.csv"}
df = load_dataset("ksgr5566/ner", revision="main", data_files = data_files)
df = pd.DataFrame(df['train']).drop(columns = ['Unnamed: 0'], axis = 0 )
df[['seed_type', 'crop_name', 'pest_name']] = df['Output'].apply(lambda x: pd.Series(extract_values(x)))
df =  df.rename(columns =  {'Input':'sentences'})
df.drop(columns=['Output'], inplace=True)
df

"""#### ***Encoding Generation***"""

def create_tags(word_token_mapping, phrase, type_agri_term ='PEST', tags = None):

    if pd.isnull(phrase):
        return(tags)
    elif phrase == '':
        return(tags)
    else :
        phrase_words = phrase.split()

        # Iterate over the word_token_mapping to find the phrase
        for i in range(len(word_token_mapping) - len(phrase_words) + 1):
            # Check if current word matches the first word of the phrase
            if word_token_mapping[i][0] == phrase_words[0]:
                match = True
                for j in range(1, len(phrase_words)):
                    if i+j >= len(word_token_mapping) or word_token_mapping[i+j][0] != phrase_words[j]:
                        match = False
                        break
                # If we found a match, tag the tokens accordingly
                if match:
                    for j, word in enumerate(phrase_words):
                        is_first_token = (j == 0)
                        for _, index in word_token_mapping[i+j][1]:
                            if is_first_token:
                                tags[index] = "B-" + type_agri_term
                                is_first_token = False
                            else:
                                tags[index] = "I-" + type_agri_term

    return (tags)

def create_word_token_mapping(sentence, tokenized_list):
    # Create a copy of the tokenized_list removing [CLS], [SEP], and [PAD], but remember their original indices
    filtered_tokens_with_indices = [(token, idx) for idx, token in enumerate(tokenized_list) if token not in ['[CLS]', '[SEP]', '[PAD]']]

    word_token_mapping = []

    for word in sentence.replace('.',' .').replace('?',' ?').split():
        current_word_tokens = []
        reconstructed_word = ''

        while filtered_tokens_with_indices and reconstructed_word != word:
            token, original_idx = filtered_tokens_with_indices.pop(0)  # Take the first token from the list
            current_word_tokens.append((token, original_idx))
            reconstructed_word += token.replace('#', '')

        if reconstructed_word != word:
            raise ValueError(f"Token mismatch for word '{word}'! Failed to reconstruct from tokens.")

        word_token_mapping.append((word, current_word_tokens))

    return word_token_mapping

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

labels = ["O", "B-CROP", "I-CROP", "B-PEST", "I-PEST", "B-SEED", "I-SEED"]
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}
default_label_id = label2id['O']

sentences = df['sentences'].apply(lambda x: [x.lower()]).to_list()
encodings = tokenizer(sentences, is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")
# encodings['input_ids'] = encodings['input_ids'].to(device)
# encodings['attention_mask'] = encodings['attention_mask'].to(device)
encodings['labels'] = torch.full_like(encodings['input_ids'], default_label_id).to(device)  # Ensure labels are also moved

for i in range(0,df.shape[0]):
    row =  df.iloc[i]
    sentence = row['sentences'].lower()
    crop_name = row['crop_name']
    pest_name =  row['pest_name']
    seed_type =  row['seed_type']

    input_id = encodings['input_ids'][i]
    tokens = tokenizer.convert_ids_to_tokens(input_id)
    word_token_mapping = create_word_token_mapping(sentence, tokens)

    tags =   ['O'] * len(tokens)
    tags = create_tags(word_token_mapping,crop_name, type_agri_term = 'CROP', tags = tags)
    tags = create_tags(word_token_mapping,pest_name, type_agri_term = 'PEST', tags = tags)
    tags = create_tags(word_token_mapping,seed_type, type_agri_term = 'SEED', tags = tags)

    attention_masks = encodings['attention_mask'][i]
    current_labels = [label2id[tag] for tag in tags] + [label2id["O"]] * (len(input_id) - len(tags))
    encodings['labels'][i] = torch.tensor(current_labels)

for i, input_id in enumerate(encodings['input_ids'][0:8]):
    tokens = tokenizer.convert_ids_to_tokens(input_id)
    labels_for_input = [id2label[label_id.item()] for label_id in encodings['labels'][i]]

    print('Original Sentence: ', ' '.join(sentences[i]))
    for token, label in zip(tokens, labels_for_input):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            print(f"{token} - {label}")
    print("----" * 10)

"""#### ***Data Preparation***"""

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].to(device) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks = train_test_split(encodings['input_ids'], encodings['labels'], encodings['attention_mask'], test_size=0.15)

# # Convert splitted data into Dataset objects
train_encodings = {'input_ids': train_inputs, 'attention_mask': train_masks, 'labels': train_labels}
val_encodings = {'input_ids': val_inputs, 'attention_mask': val_masks, 'labels': val_labels}

train_dataset = NERDataset(train_encodings)
eval_dataset = NERDataset(val_encodings)

"""#### ***Model Training***"""

# Initialize the model
model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=len(labels))
model.config.id2label = id2label
model.config.label2id = label2id

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "accuracy_score": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_total_limit=2,
    dataloader_pin_memory=False,
    # load_best_model_at_end=True,
    # save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Use train_dataset here
    eval_dataset=eval_dataset,   # Use eval_dataset here
    compute_metrics=compute_metrics,

)

trainer.add_callback(CustomCallback(trainer))

trainer.train()

"""#### ***Model Evaluation***"""

logs = trainer.state.log_history
logs = [log for log in logs if log['epoch'].is_integer()]
last_log = logs[-1]
logs = logs[:-1]
logs = pd.DataFrame(logs)
logs

# The rows of the log DataFrame are the metrics for each epoch
# go to even rows to get data of train and to odd rows to get data of eval for same epoch

# take only column starting with train as prefix and filter epocch wise
train_logs = logs[[col for col in logs.columns if col.startswith(("train", "epoch"))]]
eval_logs = logs[[col for col in logs.columns if col.startswith(("eval", "epoch"))]]

# filter even numbered rows for train and odd numbered rows for eval as per row id
train_logs = train_logs.iloc[[i for i in range(0, len(train_logs), 2)]]
eval_logs = eval_logs.iloc[[i for i in range(1, len(eval_logs), 2)]]

train_logs

eval_logs

# plot train and eval loss accuracy and f1 score curves
fig, ax = plt.subplots(2, 2, figsize=(26, 10))

ax[0, 0].plot(train_logs['epoch'], train_logs['train_loss'], label='Train Loss', color='blue')
ax[0, 0].plot(eval_logs['epoch'], eval_logs['eval_loss'], label='Eval Loss', color='red')
ax[0, 0].set_title('Train and Eval Loss')
ax[0, 0].set_xlabel('Epochs')
ax[0, 0].set_ylabel('Loss')
ax[0, 0].legend()

ax[0, 1].plot(train_logs['epoch'], train_logs['train_accuracy_score'], label='Train Accuracy', color='blue')
ax[0, 1].plot(eval_logs['epoch'], eval_logs['eval_accuracy_score'], label='Eval Accuracy', color='red')
ax[0, 1].set_title('Train and Eval Accuracy')
ax[0, 1].set_xlabel('Epochs')
ax[0, 1].set_ylabel('Accuracy')
ax[0, 1].legend()

ax[1, 0].plot(train_logs['epoch'], train_logs['train_f1'], label='Train F1 Score', color='blue')
ax[1, 0].plot(eval_logs['epoch'], eval_logs['eval_f1'], label='Eval F1 Score', color='red')
ax[1, 0].set_title('Train and Eval F1 Score')
ax[1, 0].set_xlabel('Epochs')
ax[1, 0].set_ylabel('F1 Score')
ax[1, 0].legend()

fig.delaxes(ax[1, 1])
plt.tight_layout()
plt.show()

"""#### ***Model Inference***"""

sentence = "how can i get rid of aphids on my lettuce plants?"

# tokenize the sentence
inputs = tokenizer(sentence, return_tensors="pt")
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

print("Input IDs: ", input_ids.shape)
print("Attention Mask: ", attention_mask.shape)

i = 100

input_ids = eval_dataset[i]['input_ids'].unsqueeze(0)
attention_mask = eval_dataset[i]['attention_mask'].unsqueeze(0)
labels = eval_dataset[i]['labels'].unsqueeze(0)

print("Input IDs: ", input_ids.shape)
print("Attention Mask: ", attention_mask.shape)

def infer(input_ids, attention_mask):
    # load distilbert model from saved checkpoint
    model = DistilBertForTokenClassification.from_pretrained('./results/checkpoint-10500').to(device)

    # get the model predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    predictions = outputs.logits
    predictions = predictions.detach().cpu().numpy()
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # get the predicted tags
    predicted_tags = [id2label[p] for p in predictions[0]]

    # get the tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # print the tokens and their predicted tags
    for token, tag, label in zip(tokens, predicted_tags, labels[0]):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            print(f"{token} - {tag} - {id2label[label.item()]}")
