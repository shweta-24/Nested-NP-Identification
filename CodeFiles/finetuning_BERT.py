from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import pickle5 as pickle
import numpy as np
import time
import random
import math
import torch
from torch import cuda

label_ids = {'O': 0, 'N': 1 ,'R-NP': 2, 'R': 3, 'D': 4}
MAX_LEN = 128
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 56
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
model_checkpoint = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
labels_id_map = {'O': 0, 'N': 1 ,'R-NP': 2, 'R': 3, 'D': 4}
id_labels_map = {0:'O', 1:'N', 2:'R-NP', 3:'R', 4:'D'}


device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

#Load pretrained BERT model from SentenceTransformers by HuggingFace
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label_ids))
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        # step 1: get the sentence and word labels
        sentence = self.data.tokens[index]
        tag_labels =  self.data.labels[index]
        
        # step 2: use tokenizer to encode sentence
        encoding = self.tokenizer(sentence,
                             is_split_into_words=True, 
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
        
        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [labels_id_map[tag] for tag in tag_labels]
        # create an empty array of -100 of length max_length

        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        # set only labels whose first offset position is 0 and the second is not 0
        try:
            i = 0
            for idx, mapping in enumerate(encoding["offset_mapping"]):
                if mapping[0] == 0 and mapping[1] != 0:
                # overwrite label
                    encoded_labels[idx] = labels[i]
                    i += 1
                
        except Exception as e:
            print(i)
            print(sentence)
            print(labels)
            print(tag_labels)
            print(encoded_labels)
            print(encoding["offset_mapping"])
            print()

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.LongTensor(val) for key, val in encoding.items()}
        item['labels'] = torch.LongTensor(encoded_labels)
        
        return item

    def __len__(self):
        return self.len


# Defining the training function for tuning the bert model
def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    #print(len(training_loader))
    max_ind = len(train_dataset)//165000 + 1
    print('training in ',max_ind,' iterations.')
    for ind in range(0,max_ind):
        train_dataset_batch = train_dataset.iloc[ind*165000:(ind*165000)+165000].reset_index(drop=True)
        training_loader = DataLoader(dataset(train_dataset_batch, tokenizer, MAX_LEN), **train_params)
        for idx, batch in enumerate(training_loader):
            t_start =  time.time()

            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs[0]
            tr_logits = outputs[1]
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)

            if idx % 100==0:
                loss_step = tr_loss/nb_tr_steps
                print(idx, ' ', f"Training loss per 100 training steps: {loss_step}", 'time: ',(time.time()-t_start))
                torch.cuda.empty_cache()

            # compute training accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
            #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            tr_labels.extend(labels)
            tr_preds.extend(predictions)

            tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=MAX_GRAD_NORM
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


# Defining validation loop for evaluation on test data
def valid(model, testing_loader):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            if idx == 700:
                break
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            
            #loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
            
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs[0]
            eval_logits = outputs[1]
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
        
            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            eval_labels.extend(labels)
            eval_preds.extend(predictions)
            
            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [id_labels_map[id.item()] for id in eval_labels]
    predictions = [id_labels_map[id.item()] for id in eval_preds]
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions


#Read the dataset created by dataloaderds    
dataset_file_path = 'DataFiles/model_dataset_alphaLablesFiltered.pkl'
with open(dataset_file_path, "rb") as dt:
    raw_dataset = pickle.load(dt)

#Divide dataset into training and testing data (80% - 20%)
raw_dataset.reset_index(inplace=True)
sentence_ids_list = list(raw_dataset['sentence_id'].unique())
random.seed(42)
train_size = 0.8
selected_ids = random.sample(sentence_ids_list, math.floor(train_size*len(sentence_ids_list)))

train_dataset = raw_dataset[raw_dataset['sentence_id'].isin(selected_ids)]
test_dataset = raw_dataset.drop(train_dataset.index).reset_index(drop=True)
train_dataset.reset_index(drop=True, inplace=True)

#print(train_dataset)
#print('-------------')
#print(test_dataset)
test_dataset.to_pickle('DataFiles/test_dataset.pkl')  #Save test data for evaluation

print("FULL Dataset: {}".format(raw_dataset.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = dataset(train_dataset, tokenizer, MAX_LEN)
testing_set = dataset(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

# Create dataloaders for training loop
training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


#Call the training loop
for epoch in range(EPOCHS):
    print(f"Training epoch: {epoch + 1}")
    train(epoch)

# Evaluating the model on test data
labels, predictions = valid(model, testing_loader)


#save the fine-tuned model
if device != 'cpu':
    modelnamegpu = 'Models/FineTunedBERTGPU_OntoNotes'
    torch.save(model.state_dict(), modelnamegpu)

model.to('cpu')
modelnamecpu = 'Models/FineTunedBERTCPU_OntoNotes'
torch.save(model.state_dict(), modelnamecpu)

print('Model saved')

    

