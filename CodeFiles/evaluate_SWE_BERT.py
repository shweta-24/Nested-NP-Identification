
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import pickle
import numpy as np
import tikzplotlib


class SweBERTTokenClassifier(nn.Module):
    def __init__(self,model_checkpoint,num_labels): 
        super(SweBERTTokenClassifier,self).__init__() 
        self.num_labels = num_labels 

        #Load Model with given checkpoint and extract its body
        self.model = model = AutoModel.from_pretrained(model_checkpoint,config=AutoConfig.from_pretrained(model_checkpoint, output_attentions=True,output_hidden_states=True))
        self.dropout = nn.Dropout(0.1) 
        self.classifier = nn.Linear(768,num_labels) # load and initialize weights

    def forward(self, input_ids=None, attention_mask=None,labels=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        #Add custom layers
        sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state
        logits = self.classifier(sequence_output) # calculate losses
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)


labels_id_map = {'O': 0, 'N': 1 ,'R-NP': 2, 'R': 3, 'D': 4}
id_labels_map = {0:'O', 1:'N', 2:'R-NP', 3:'R', 4:'D'}
MAX_LEN = 128

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)



tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
if device == 'cpu':
    model = torch.load('./Models/SWEBERTModelCPU_FineTuned')
else:
    model = torch.load('./Models/SWEBERTModelGPU_FineTuned')
model.to(device)
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

def store_NP_tree(parse):
    NP_list = {}
    for ind, row in parse.iterrows():
        sentence = row['tokens']
        prediction = row['labels']
        ind_word_map = {}
        word_count = 0
        for i, t in enumerate(sentence):
            if t != '|':
                sentence[i] = str(word_count)
                ind_word_map[str(word_count)] = t
                word_count += 1

        prev_sep_ind = 0
        in_rnp = False
        for ind, word in enumerate(prediction):
            if word == 'N':
                if 1 not in list(NP_list):
                    NP_list[1] = []
                if sentence[ind] not in NP_list[1]:
                    NP_list[1].append(sentence[ind])
            elif word == 'D':
                if in_rnp:
                    s = (' '.join(sentence[prev_sep_ind:ind+1])).replace('| ', '').replace(' |','')
                    if len(s.split()) not in list(NP_list):
                        NP_list[len(s.split())] = []
                    if s not in NP_list[len(s.split())]:
                        NP_list[len(s.split())].append(s)
                    in_rnp = False
                prev_sep_ind = ind
            elif word == 'R-NP':
                in_rnp = True
    return NP_list, ind_word_map

def convert_num2words(NP_list, ind_word_map):
    for phrase_len, phrase_list in NP_list.items():
        for ind, phrase in enumerate(phrase_list):
            s = []
            for n in phrase.split():
                s.append(ind_word_map[n])
            NP_list[phrase_len][ind] = ' '.join(s)
    return NP_list


def calculate_recall(true_parse, infered_parse):
    total_phrases = 0
    recalled_phrases_seq = 0
    recalled_phrases_subtree = 0
    recall_tree = 1
    missed_phrases = []

    for phrase_len_r in list(infered_parse):
        if phrase_len_r not in list(true_parse):
            for phrase_r in infered_parse[phrase_len_r]:
                missed_phrases.append(phrase_r)

    for phrase_len in list(true_parse):
        total_phrases += len(true_parse[phrase_len])
        if phrase_len in list(infered_parse):
            for phrase in true_parse[phrase_len]:
                if phrase in infered_parse[phrase_len]:
                    recalled_phrases_seq += 1
                    found_missed_flag =  False
                    for p in missed_phrases:
                        if p in phrase:
                            found_missed_flag = True
                    if found_missed_flag == False:
                        recalled_phrases_subtree += 1             
                else:
                    #recall_tree = 0
                    missed_phrases.append(phrase)
        else:
            #recall_tree = 0
            for p in true_parse[phrase_len]:
                missed_phrases.append(p)
    if len(missed_phrases) > 0:
        recall_tree = 0
    else:
        recall_tree = 1

    
    return {'total_phrases':total_phrases, 'recall_seq':recalled_phrases_seq, 'recall_subtree':recalled_phrases_subtree, 'recall_tree': recall_tree}


def calculate_precision(true_parse, infered_parse):
    total_phrases = 0
    precise_phrases_seq = 0
    precise_phrases_subtree = 0
    missed_phrases = []

    for phrase_len_r in list(true_parse):
        if phrase_len_r not in list(infered_parse):
            for phrase_r in true_parse[phrase_len_r]:
                missed_phrases.append(phrase_r)
 
    for phrase_len in list(infered_parse):
        total_phrases += len(infered_parse[phrase_len])
        if phrase_len in list(true_parse):
            for phrase in infered_parse[phrase_len]:
                if phrase in true_parse[phrase_len]:
                    precise_phrases_seq += 1
                    found_missed_flag =  False
                    for p in missed_phrases:
                        if p in phrase:
                            found_missed_flag = True
                    if found_missed_flag == False:
                        precise_phrases_subtree += 1             
                else:
                    #precise_tree = 0
                    missed_phrases.append(phrase)
        else:
            #precise_tree = 0
            for p in infered_parse[phrase_len]:
                missed_phrases.append(p)
    if len(missed_phrases) > 0:
        precise_tree = 0
    else:
        precise_tree = 1

    return {'total_phrases':total_phrases, 'precise_seq':precise_phrases_seq, 'precise_subtree':precise_phrases_subtree, 'precise_tree': precise_tree}


def calculate_tree_depth(parse):
    tracking_phrase = {}
    for ix, l in enumerate(list(parse)):
        if ix == 0: 
            for p in parse[l]:
                tracking_phrase[p] = 1
        else:
            tracking_phrase_prev = tracking_phrase.copy()
            for p in list(tracking_phrase_prev):
                for q in parse[l]:
                    if p in q:
                        tracking_phrase[q] = tracking_phrase[p]+1
                        tracking_phrase.pop(p)
                        break
            for n in parse[l]:
                if n not in tracking_phrase:
                    tracking_phrase[n] = 1
    if len(tracking_phrase) == 0:
        max_depth = 0
    else:
        max_depth = max(list(tracking_phrase.values()))
    return max_depth

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def evaluate():
    dataset_file_path = 'DataFiles/model_dataset_swetreebank_filtered_test_v1.pkl'

    with open(dataset_file_path, "rb") as dt:
        test_dataset = pickle.load(dt)
    test_dataset.reset_index()

    sentence_ids_test_list = list(test_dataset['sentence_id'].unique())

    #ex_list = [73, 1120, 2048]
    #ex_list = [2, 5]

    true_parse_all = pd.DataFrame(columns=['sentence_id', 'sentence', 'tokens', 'labels'])
    infered_parse_all = pd.DataFrame(columns=['sentence', 'tokens', 'labels'])

    metrics_file = 'DataFiles/metrics_swetreebank.pkl'

    if os.path.exists(metrics_file):
        metrics_all = {}
        with open(metrics_file, 'rb') as f:
            metrics_all = pickle.load(f)
        start_idx = max(list(metrics_all))
    else:
        metrics_all = {}
        start_idx = 0

    t_start = time.time()
    print('Calucating metrics for ',len(sentence_ids_test_list),' test cases.')

    for ex_num in range(start_idx, len(sentence_ids_test_list)):
    #for ix, ex_num in enumerate(ex_list):
        try:
            ex = sentence_ids_test_list[ex_num]

            ex_parse = test_dataset[test_dataset['sentence_id'] == ex]
            sent = ex_parse.iloc[0]['sentence'].replace(' |', '').replace('| ','')
            infered_parse = infer_sentence_parse(list(ex_parse['tokens'])[0])

            true_parse_all = true_parse_all.append(ex_parse, ignore_index=True)
            infered_parse_all = infered_parse_all.append(infered_parse, ignore_index=True)

            NP_list_true, ind2word_true = store_NP_tree(ex_parse)
            NP_list_infered, ind2word_infered = store_NP_tree(infered_parse)

            tree_depth  = calculate_tree_depth(NP_list_true)
            recall = calculate_recall(NP_list_true, NP_list_infered)
            precision = calculate_precision(NP_list_true, NP_list_infered)
            NP_list_infered_words = convert_num2words(NP_list_infered, ind2word_infered)
            NP_list_true_words = convert_num2words(NP_list_true, ind2word_true)
            NP_true_count = {ph_len:len(np_list) for ph_len,np_list in NP_list_true_words.items()}
            NP_infered_count = {ph_len:len(np_list) for ph_len,np_list in NP_list_infered_words.items()}
            metrics_all[ex_num] = {'sentence':sent, 'recall':recall, 'precision':precision, 'NP_true_count':NP_true_count, 'NP_infered_count':NP_infered_count, 'tree_depth':tree_depth}
            if ex_num%100 == 0:
                print('sentence ',ex_num,' : ',sent)
                print(metrics_all[ex_num])
                print('Time so far: ',(time.time()-t_start),' secs')
                print()

                with open(metrics_file, 'wb') as f:
                    pickle.dump(metrics_all, f)

            
        except Exception as e:
            print(e)
            print('Couldn\'t generate parse for ', sent )   

    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics_all, f)
    true_parse_all.to_csv('Data/res_swe.csv')
    infered_parse_all.to_csv('Data/res_inf_swe.csv')
    print('Total Time to calculate metrics: ',(time.time()-t_start),' secs')


    print('Actual test Cases',len(metrics_all))
    recall_list = []
    recall_list_subtree = []
    recall_list_tree = []
    precision_list = []
    precision_list_subtree = []
    precision_list_tree = []
    f1_score_list = []
    f1_score_list_subtree = []
    f1_score_list_tree = []
    recall_for_plot = {'seq':{},'subtree':{},'tree':{}}
    precision_for_plot = {'seq':{},'subtree':{},'tree':{}}

    for test_case_num in list(metrics_all):
        #sent_len = len(metrics_all[test_case_num]['sentence'].split()) #Uncomment to calculate metrics along sentence length
        sent_len = metrics_all[test_case_num]['tree_depth']
        if sent_len not in recall_for_plot['seq']:
            recall_for_plot['seq'][sent_len] = []
            recall_for_plot['subtree'][sent_len] = []
            recall_for_plot['tree'][sent_len] = []

            precision_for_plot['seq'][sent_len] = []
            precision_for_plot['subtree'][sent_len] = []
            precision_for_plot['tree'][sent_len] = []

        got_both = False
        if metrics_all[test_case_num]['recall']['total_phrases'] != 0:
            got_both = True
            recall_list.append(metrics_all[test_case_num]['recall']['recall_seq']/metrics_all[test_case_num]['recall']['total_phrases'])
            recall_list_subtree.append(metrics_all[test_case_num]['recall']['recall_subtree']/metrics_all[test_case_num]['recall']['total_phrases'])
            recall_list_tree.append(metrics_all[test_case_num]['recall']['recall_tree'])
            recall_for_plot['seq'][sent_len].append(recall_list[len(recall_list)-1])
            recall_for_plot['subtree'][sent_len].append(recall_list_subtree[len(recall_list_subtree)-1])
            recall_for_plot['tree'][sent_len].append(recall_list_tree[len(recall_list_tree)-1])

        if metrics_all[test_case_num]['precision']['total_phrases'] != 0:
            precision_list.append(metrics_all[test_case_num]['precision']['precise_seq']/metrics_all[test_case_num]['precision']['total_phrases'])
            precision_list_subtree.append(metrics_all[test_case_num]['precision']['precise_subtree']/metrics_all[test_case_num]['precision']['total_phrases'])
            precision_list_tree.append(metrics_all[test_case_num]['precision']['precise_tree'])
            precision_for_plot['seq'][sent_len].append(precision_list[len(precision_list)-1])
            precision_for_plot['subtree'][sent_len].append(precision_list_subtree[len(precision_list_subtree)-1])
            precision_for_plot['tree'][sent_len].append(precision_list_tree[len(precision_list_tree)-1])
        else:    
            if got_both:
                got_both = False
        
        if got_both:
            if recall_list[-1]+precision_list[-1] != 0:
                f1_score_list.append((2*recall_list[-1]*precision_list[-1])/(recall_list[-1]+precision_list[-1]))
            else:
                f1_score_list.append(0)
            if recall_list_subtree[-1]+precision_list_subtree[-1] != 0:
                f1_score_list_subtree.append((2*recall_list_subtree[-1]*precision_list_subtree[-1])/(recall_list_subtree[-1]+precision_list_subtree[-1]))
            else:
                f1_score_list_subtree.append(0)
            if recall_list_tree[-1]+precision_list_tree[-1] != 0:
                f1_score_list_tree.append((2*recall_list_tree[-1]*precision_list_tree[-1])/(recall_list_tree[-1]+precision_list_tree[-1]))
            else:
                f1_score_list_tree.append(0)

    print('Recall Sequence Level: ',round(np.average(recall_list)*100, 2), '%')
    print('Recall Subtree Level: ',round(np.average(recall_list_subtree)*100, 2), '%')
    print('Recall Tree Level: ',round((np.sum(recall_list_tree)/len(recall_list_tree))*100, 2), '%')
    print()
    print('Precision Sequence Level: ',round(np.average(precision_list)*100, 2), '%')
    print('Precision Subtree Level: ',round(np.average(precision_list_subtree)*100, 2), '%')
    print('Precise Tree Level: ',round((np.sum(precision_list_tree)/len(precision_list_tree))*100, 2), '%')
    print()
    print('F1 Score Sequence Level: ',round(np.average(f1_score_list)*100, 2), '%')
    print('F1 Score Subtree Level: ',round(np.average(f1_score_list_subtree)*100, 2), '%')
    print('F1 Score Tree Level: ',round((np.average(f1_score_list_tree))*100, 2), '%')

    y_labels = {'seq':'Sequence Level', 'subtree':'Subtree Level', 'tree':'Tree Level'}


    for i, key in enumerate(list(recall_for_plot)):
        recall_plot_dict = dict(sorted(recall_for_plot[key].items()))
        plot_points = {}
        for k in list(recall_plot_dict):
            plot_points[k] = (round(np.average(recall_plot_dict[k]),4), len(recall_plot_dict[k]))
        print(plot_points)
        print()
        plt.plot(list(plot_points.keys()), [i[0] for i in list(plot_points.values())], label = list(y_labels.values())[i], marker = 'o')
    plt.xlabel('Maximal depth of nested NPs')   # Update if plotting across sentence length
    plt.ylabel('Recall')    # Update if plotting across sentence length

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True)
    fig = plt.gcf()
    tikzplotlib_fix_ncols(fig)
    plt.savefig('Results/Swe/Recallvsdepth.png')
    tikzplotlib.save("Results/Swe/Recalldepth.tex")
    plt.show()


    for i, key in enumerate(precision_for_plot):
        precision_plot_dict = dict(sorted(precision_for_plot[key].items()))
        plot_points = {}
        for k in list(precision_plot_dict):
            plot_points[k] = (round(np.average(precision_plot_dict[k]),4), len(precision_plot_dict[k]))
        print(plot_points)
        print()
        plt.plot(list(plot_points.keys()), [i[0] for i in list(plot_points.values())], label = list(y_labels.values())[i], marker = 'o')

    plt.xlabel('Maximal depth of nested NPs')   # Update if plotting across sentence length
    plt.ylabel('Precision')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True)
    fig = plt.gcf()
    tikzplotlib_fix_ncols(fig)
    plt.savefig('Results/Swe/Precisionvsdepth.png')
    tikzplotlib.save("Results/Swe/Precisiondepth.tex")
    plt.show()




if __name__ == '__main__':

    sentences = ['På sikt går nog sjön mot sin undergång .']
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
    
    evaluate()


    
    


