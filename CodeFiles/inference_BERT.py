
from transformers import BertTokenizerFast, BertForTokenClassification
import torch
import pandas as pd

labels_id_map = {'O': 0, 'N': 1 ,'R-NP': 2, 'R': 3, 'D': 4}
id_labels_map = {0:'O', 1:'N', 2:'R-NP', 3:'R', 4:'D'}
MAX_LEN = 128


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(labels_id_map))
model.to(device)

### Select trained model for inference
if device == 'cpu':
    modelpath = 'Models/FineTunedBERTCPU_OntoNotes'
else:
    modelpath = 'Models/FineTunedBERTGPU_OntoNotes'
model.load_state_dict(torch.load(modelpath))
model.eval()


def store_NP(sentence, prediction):
    NP_list = []
    prev_sep_ind = 0
    in_rnp = False
    for ind, word in enumerate(prediction):
        if word == 'N':
            if word not in NP_list:
                NP_list.append(sentence[ind])
        elif word == 'D':
            if in_rnp:
                s = (' '.join(sentence[prev_sep_ind:ind+1])).replace('| ', '').replace(' |','')
                if s not in NP_list:
                    NP_list.append(s)
                in_rnp = False
            prev_sep_ind = ind
        elif word == 'R-NP':
            in_rnp = True

    return NP_list

def infer_sentence_parse(sent):
    parse_list = []
    sentence = ' '.join(sent)
    old_predictions = None
    prediction = []
    it = 0
    while True:
        inputs = tokenizer(sentence.split(),
                            is_split_into_words=True, 
                            return_offsets_mapping=True, 
                            padding='max_length', 
                            truncation=True, 
                            max_length=MAX_LEN)

        # move to gpu
        ids = torch.LongTensor([inputs["input_ids"]]).to(device)
        mask = torch.LongTensor([inputs["attention_mask"]]).to(device)
        # forward pass
        outputs = model(ids, attention_mask=mask)
        logits = outputs[0]

        active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [id_labels_map[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)
        old_predictions = prediction
        prediction = []
        for token_pred, mapping in zip(wp_preds, torch.LongTensor(inputs["offset_mapping"]).squeeze().tolist()):
        #only predictions on first word pieces are important
            if mapping[0] == 0 and mapping[1] != 0:
                prediction.append(token_pred[1])
            else:
                continue
        if old_predictions == prediction:
            break
        #print(sentence.split())
        #print(prediction)
        #print()
        parse_list.append({'sentence':sentence, 'tokens':sentence.split(), 'labels':prediction})
        sentence = ' '.join([sentence.split()[i] for i in range (len(sentence.split())) if prediction[i] not in ('R', 'R-NP')])
        it = it + 1 
    return pd.DataFrame(parse_list)

if __name__ == '__main__':

    sentences = ['The house has a window .', 'The house has a window with a wooden frame .', 
               'The house in the long street has a window with a wooden frame .']

    for sent in sentences:
        NP_list_complete = []
        sentence = ' | '.join(sent.split())
        #sentence = "The | house | has | a | window | with | a | wooden | frame | ."
        old_predictions = None
        prediction = []
        while True:
            inputs = tokenizer(sentence.split(),
                                is_split_into_words=True, 
                                return_offsets_mapping=True, 
                                padding='max_length', 
                                truncation=True, 
                                max_length=MAX_LEN)

            # move to gpu
            ids = torch.LongTensor([inputs["input_ids"]]).to(device)
            mask = torch.LongTensor([inputs["attention_mask"]]).to(device)
            # forward pass
            outputs = model(ids, attention_mask=mask)
            logits = outputs[0]

            active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

            tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
            token_predictions = [id_labels_map[i] for i in flattened_predictions.cpu().numpy()]
            wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)
            old_predictions = prediction
            prediction = []
            for token_pred, mapping in zip(wp_preds, torch.LongTensor(inputs["offset_mapping"]).squeeze().tolist()):
            #only predictions on first word pieces are important
                if mapping[0] == 0 and mapping[1] != 0:
                    prediction.append(token_pred[1])
                else:
                    continue
            if old_predictions == prediction:
                break
            print(sentence.split())
            print(prediction)
            print()
            NP_list = store_NP(sentence.split(), prediction)
            NP_list_complete = list(set(NP_list_complete + NP_list))
            sentence = ' '.join([sentence.split()[i] for i in range (len(sentence.split())) if prediction[i] not in ('R', 'R-NP')])
        print(NP_list_complete)
        print('---------------')
        print()
    
    


