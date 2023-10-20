import json
from collections import defaultdict
import copy
import os
import pickle
import sys
import pandas as pd

'''
store datapoints into pickle file
Input:
    path: string - path of the data file
    result: dict - sentence to token labels mapping (each entry is a training datapoint for the model)

Output:
    None
'''
def store_datapoints(path, result):
    with open(path, 'wb') as f:
        pickle.dump(result, f)
    #print('stored all datapoints for this text')


'''
read datapoints from pickle file into memory
Input:
    path: string - path of the data file
    
Output:
    result: dict - sentence to token labels mapping (each entry is a training datapoint for the model)

'''
def read_datapoints(path):
    with open(path, 'rb') as f:
        result = pickle.load(f)
    return result


'''
Creating all possible continuos phrases of length n contained in a phrase
Input:
    token_list: list - list of indices(word representations of sentence)
    n: integer - desired length of n_gram

Output:
    ngram_list: list - list of all possible continuos n_grams for the token_list

'''
def create_token_ngrams(token_list, n):
    ngram_list =[]
    for i in range(len(token_list)-n+1):
        ngram_list.append(' '.join(token_list[i:i+n]))

    return ngram_list


'''
Convert indices back to words and clean-up decision labels 
Input:
    wordlist: list - list of indices(word representations of sentence)
    sent: string - sentence string with index representations of words
    decisions: list - list of labels for each token in the sentence

Output:
    str_word_sent: string - sentence string with words replacing index representations of words
    stand_decisions: list - list of cleaned-up decision labels

'''
def convert_to_standard(wordlist, sent, decisions):
    word_sent = []
    stand_decisions = []
    for ch in sent.split():
        if ch == '|':
            word_sent.append(ch)
        else:
            word_sent.append(wordlist[int(ch)])
    for ch in decisions:
        if ch in ['N', 'O']:
            stand_decisions.append(ch)
        elif 'R-NP' in ch:
            stand_decisions.append('R-NP')
        elif 'R-' in ch:
            stand_decisions.append('R')
        elif 'D-' in ch:
            stand_decisions.append('D')
        else:
            print('There is an alien word in the decisions list: ',ch)
            sys.exit(0)
    
    str_word_sent = ' '.join(word_sent)

    return str_word_sent, stand_decisions


'''
Convert indices back to words and clean-up decision labels 
Input:
    wordlist: list - list of indices(word representations of sentence)
    sent: string - sentence string with index representations of words
    decisions: list - list of labels for each token in the sentence

Output:
    str_word_sent: string - sentence string with words replacing index representations of words
    stand_decisions: list - list of cleaned-up decision labels

'''
def create_training_datapoints(word_list, phrase_tag_map, word_tag_map):
    results = {}
    sent  = ' | '.join([str(i) for i in range(len(word_list))])
    phrase_lens = list(phrase_tag_map)
    prev_sent = []
    for phrase_len in phrase_lens:
        if phrase_len < 2:
            continue
        range_to_tag = {}
        new_sent = []
        decisions = []
        i = 0
        while(i < len(word_list)):
            for tag in phrase_tag_map[phrase_len]:
                for rang in phrase_tag_map[phrase_len][tag]:
                    range_to_tag[rang[0]] = tag
            
            if i in list(range_to_tag):
                range_str = ''
                tag = range_to_tag[i]
                for n in range(i,i+phrase_len):
                    if word_tag_map[n] == 'NP':  
                        decisions.append('N')
                    else:
                        decisions.append('O')
                    if n != i+phrase_len-1:
                        range_str = range_str + str(n) + ' '
                        if tag == 'NP':
                            decisions.append('R-NP'+'-'+str(n))
                        else:
                            decisions.append('R'+'-'+str(n))
                    else:
                        range_str = range_str + str(n)
                        decisions.append('D'+'-'+str(n))
                    
                new_sent.append(range_str)
                i = i + phrase_len
            else:
                new_sent.append(str(i))
                if word_tag_map[i] == 'NP':  
                    decisions.append('N')
                else:
                    decisions.append('O')

                if i != len(word_list)-1:
                    decisions.append('D'+'-'+str(i))
                i = i + 1

        if len(prev_sent) > 0:
            ss = ''
            str_ix = 0
            s_list = copy.copy(new_sent)
            read_s = 0
            for s in s_list:
                read_s += 1
                if ss == '':
                    ss = ss + s
                else:
                    ss = ss + ' ' + s

                if ss in prev_sent:
                    new_sent[str_ix] = ss

                    for n in range(str_ix+1, str_ix+read_s): 
                        del new_sent[str_ix+1]
                    for idx, n in enumerate(ss.split()):
                        if idx < len(ss.split())-1:
                            try:
                                decisions.remove('D-'+n)
                            except ValueError:
                                pass
                            try:
                                decisions.remove('R-'+n)
                            except ValueError:
                                pass
                            try:
                                decisions.remove('R-NP-'+n)
                            except ValueError:
                                pass

                    str_ix += 1
                    ss = ''
                    read_s = 0
                else:
                    new_ss_list = ss.split()
                    if len(new_ss_list) == phrase_len:
                        str_ix += 1
                        if phrase_len-1 > 1:
                            for token_len in range(phrase_len-1, 1, -1):
                                new_ss_grams = create_token_ngrams(new_ss_list,token_len)
                                for new_ss_gram in new_ss_grams:
                                    if new_ss_gram in prev_sent:
                                        for idx, n in enumerate(new_ss_gram.split()):
                                            if idx < len(new_ss_gram.split())-1:
                                                try:
                                                    decisions.remove('D-'+n)
                                                except ValueError:
                                                    pass
                                                try:
                                                    decisions.remove('R-'+n)
                                                except ValueError:
                                                    pass
                                                try:
                                                    decisions.remove('R-NP-'+n)
                                                except ValueError:
                                                    pass 
                        ss = ''
                        read_s = 0


        prev_sent = new_sent
        
        word_sent, standard_decisions = convert_to_standard(word_list, sent, decisions)
        results[word_sent] = standard_decisions
        sent = ' | '.join(map(str,new_sent))
    return results


'''
Create dataframe from datapoints
Input:
    data: dict - Mappping of sentence id to all possible datapoints (sentence to token_labels mappings) for that sentence 
    
Output:
    dataset: DataFrame - dataframe with columns ['sentence_id', 'sentence', 'tokens', 'labels']

'''
def create_dataset(data, label_ids):
    dataset = []
    print('reached create dataset')
    for sentence_idx in list(data):
        for sentence in list(data[sentence_idx]):
            #label_id_list = list(map(label_ids.get, data[sentence_idx][sentence]))
            dataset.append({'sentence_id':sentence_idx, 'sentence':sentence, 'tokens':sentence.split(), 'labels':data[sentence_idx][sentence]})
    print('Dataframe created')
    print(pd.DataFrame(dataset))
    return pd.DataFrame(dataset)


if __name__ == '__main__':
    files = [('DataFiles/swedish_treebank/suc/suc-test.parse', 'DataFiles/processed_datapoints_suctest_v1.pkl'),
            ('DataFiles/swedish_treebank/talbanken/talbanken-test.parse', 'DataFiles/processed_datapoints_talbankentest_v1.pkl')]

    noun_tags = ['NN','PM', 'PN','NP']

    all_datapoints = []
    for (f1, f2) in files:
        f = f1
        datapoints_file_path = f2
        file1 = open(f, 'r', encoding='utf8')
        lines = file1.readlines()
        end_idx = len(lines)
        fc = 0

        if os.path.exists(datapoints_file_path):
            datapoints = read_datapoints(datapoints_file_path)
            start_idx = max(list(datapoints))
        else:
            datapoints = {}
            start_idx = 0
        for ex in range(start_idx, end_idx):
            try:
                #print('----------------')
                word_tag_map = {}
                word_index_map = {}
                index_word_map = {}
                word_list = []
                line = lines[ex]
                line = line.replace(')',' )')
                line = line.replace('(','( ')
                #print(line)
                open_idx = []
                word_idx = -1
                
                for ix, ch in enumerate(line): 
                    if ch == '(':
                        open_idx.append(ix)
                    elif ch == ')':
                        open_ix = open_idx.pop()
                        split_tag_word = line[open_ix+1:ix].split()
                        tag = split_tag_word[0]
                        if tag in noun_tags:
                            tag = 'NP'
                        word = split_tag_word[-1]
                        if ')' not in word:
                            word_idx += 1
                            #word_tag_map[word.strip()] = tag.strip()
                            word_list.append(word.strip())
                            word_tag_map[word_idx] = tag.strip()
                            index_word_map[word_idx] = word.strip()
                            if word in list(word_index_map):
                                word_index_map[word.strip()].append(word_idx)
                            else:
                                word_index_map[word] = [word_idx]
                
                line_word_index = ''
                for ch in line.split():
                    if ch in list(word_index_map):
                        i = word_index_map[ch].pop(0)
                        line_word_index += ' '+str(i)+' '
                    else:
                        line_word_index += ch+' '
                line_word_index = line_word_index.strip()

                open_idx = []
                phrase_tag_map = {}
                for ix, ch in enumerate(line_word_index):
                    if ch == '(':
                        open_idx.append(ix)
                    elif ch == ')':
                        open_ix = open_idx.pop()
                        split_tag_word = line_word_index[open_ix+1:ix].split()
                        tag = split_tag_word[0].strip()
                        if tag in noun_tags:
                            tag = 'NP'
                        phrase_str = [x.strip() for x in split_tag_word[1:] if x.strip().isnumeric()]
                        phrase = (int(phrase_str[0]),int(phrase_str[-1]))
                        if tag in list(phrase_tag_map):
                            phrase_tag_map[tag].append(phrase)
                        else:
                            phrase_tag_map[tag] = [phrase]     

                phrase_tag_len_map = defaultdict(lambda: defaultdict(lambda: []))
                for tag in list(phrase_tag_map):
                    if tag == 'ROOT':
                        continue
                    for phrase in phrase_tag_map[tag]:
                        phrase_len = phrase[1]-phrase[0]+1
                        if phrase_len == 1:
                            continue
                        phrase_tag_len_map[phrase_len][tag].append(phrase)

                results = create_training_datapoints(word_list, phrase_tag_len_map, word_tag_map)
                datapoints[ex] = results
            except Exception as e:
                fc += 1
                print('Could not create datapoint : ',ex,'. Failure count ',fc)
                print(line)

        store_datapoints(datapoints_file_path, datapoints)
        print(len(datapoints))
        print(datapoints[10])
        all_datapoints.append(datapoints)


    label_ids = {'O': 0, 'N': 1 ,'R-NP': 2, 'R': 3, 'D': 4}

    dataset = pd.DataFrame()
    for i, dp in enumerate(all_datapoints):
        dataset = dataset.append(create_dataset(dp, label_ids), ignore_index=True)

    dataset.reset_index()
    print(dataset)

    dataset_file_path = 'DataFiles/model_dataset_swetreebank_test_v1.pkl'
    dataset.to_pickle(dataset_file_path)

    filtered_dataset_file_path = 'DataFiles/model_dataset_swetreebank_filtered_test_v1.pkl'
    filtered_dataset = dataset[dataset.tokens.map(len) == dataset.labels.map(len)].reset_index()
    print(filtered_dataset)
    filtered_dataset.to_pickle(filtered_dataset_file_path)


