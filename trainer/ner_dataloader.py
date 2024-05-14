import pandas as pd
from transformers import DistilBertTokenizerFast
import torch
import ast


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].to(device) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])
    

class NERDataLoader:
    def __init__(self, file_path='../dataset/train.csv'):
        self.file_path = file_path
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.labels = ["O", "B-CROP", "I-CROP", "B-PEST", "I-PEST", "B-SEED", "I-SEED"]

        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.default_label_id = self.label2id['O']

    def __extract_values(self, output_str):
        output_dict = ast.literal_eval(output_str)
        seed_type = output_dict.get('seed_type', None)
        crop_name = output_dict.get('crop_name', None)
        pest_name = output_dict.get('pest_name',  None)
        return seed_type, crop_name, pest_name
    
    def __make_dataframe(self, file_path):
        data = pd.read_csv(file_path).drop(columns = ['Unnamed: 0'], axis = 0)
        data[['seed_type', 'crop_name', 'pest_name']] = data['Output'].apply(lambda x: pd.Series(self.__extract_values(x)))
        data = data.rename(columns =  {'Input':'sentences'})
        data.drop(columns=['Output'], inplace=True)

        return data
    
    def create_tags(self, word_token_mapping, phrase, type_agri_term ='PEST', tags = None):
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
    
    def create_word_token_mapping(self, sentence, tokenized_list):
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
    
    def make_encodings(self, data):
        sentences = data['sentences'].apply(lambda x: [x.lower()]).to_list()

        encodings = self.tokenizer(sentences, is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")
        encodings['labels'] = torch.full_like(encodings['input_ids'], self.default_label_id).to(device)
        
        tags = []
        for i in range(0, data.shape[0]):
            row =  data.iloc[i]
            sentence = row['sentences'].lower()
            crop_name = row['crop_name']
            pest_name =  row['pest_name']
            seed_type =  row['seed_type']

            input_id = encodings['input_ids'][i]
            tokens = self.tokenizer.convert_ids_to_tokens(input_id)
            word_token_mapping = self.create_word_token_mapping(sentence, tokens)

            tags =   ['O'] * len(tokens)
            tags = self.create_tags(word_token_mapping, crop_name, type_agri_term = 'CROP', tags = tags)
            tags = self.create_tags(word_token_mapping, pest_name, type_agri_term = 'PEST', tags = tags)
            tags = self.create_tags(word_token_mapping, seed_type, type_agri_term = 'SEED', tags = tags)

            attention_masks = encodings['attention_mask'][i]
            current_labels = [self.label2id[tag] for tag in tags] + [self.label2id["O"]] * (len(input_id) - len(tags))
            encodings['labels'][i] = torch.tensor(current_labels)

        return encodings
    
    def get_ner_dataset(self):
        data = self.__make_dataframe(self.file_path)
        encodings = self.make_encodings(data)

        ner_dataset = NERDataset(encodings)

        return ner_dataset
    