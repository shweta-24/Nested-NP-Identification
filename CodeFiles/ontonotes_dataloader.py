
'''
Convert OntoNotes corpus to json file using the following command.

py -3.10-64 ontonotes5_to_json.py -s DataFiles/ontonotes-release-5.0_LDC2013T19.tgz 
-d DataFiles/ontonotes5.json -r 42
'''

import json
import copy
import sys
import pickle
import os
import pandas as pd
import time

noun_tags = ['NN','NNS','NNP','NNPS','NP','NS']

'''
Creating word to position index mapping in sentence 
Input:
    text:string - sentence string
    morphology: dict - POS tag to word mapping for each word in sentence

Output:
    word_pos_map: dict - index to word mapping in a sentence. Index to be used as representation of the word for further processing
    word_lis: list  - list of all the words in text

'''
def create_word_pos_map(text, morphology):
    word_list = []
    word_cuts = []
    word_pos_map = {}
    for tag in list(morphology):
        for indRange in morphology[tag]:
            word_cuts.append(indRange[0])
    word_cuts.sort()

    for idx, ind in enumerate(word_cuts):
        if idx < len(word_cuts)-1:
            if text[word_cuts[idx+1]-1] != ' ':
                word_list.append(text[ind:word_cuts[idx+1]])
                word_pos_map[idx] = [ind, word_cuts[idx+1]]
            else:
                word_list.append(text[ind:word_cuts[idx+1]-1])
                word_pos_map[idx] = [ind, word_cuts[idx+1]-1]
        else:
            word_pos_map[idx] = [word_cuts[idx], len(text)]
    

    word_list.append(text[ind:])
    
    return word_pos_map, word_list


'''
Creating word index to POS tag mapping in sentence
Input:
    word_pos_map: dict - index to word mapping in a sentence.
    morphology: dict - POS tag to word mapping for each word in sentence

Output:
    word_tag_map: dict - word to index mapping in a sentence, for reverse reference of index

'''
def create_word_tag_map(word_pos_map, morphology):

    word_tag_map = {}
    for tag in list(morphology):
        if tag in noun_tags or tag.split('-')[0] in noun_tags:
            tag_to_assign = 'NP'
        else:
            tag_to_assign = tag 
        for indRange in morphology[tag]:
            word = list(word_pos_map.values()).index(indRange)
            word_tag_map[word] = tag_to_assign
    return word_tag_map
        

'''
Creating phrase to POS tag mapping in sentence
Input:
    word_pos_map: dict - index to word mapping in a sentence.
    syntax: dict - POS tag to character index mapping for each phrase in the sentence

Output:
    phrase_tag_map: dict - POS tag to word index mapping in the sentence. Sorted by phrase length

'''        
def create_phrase_tag_map(word_pos_map, syntax):
    phrase_tag_map = {}
    for tag in list(syntax):
        if tag in noun_tags or tag.split('-')[0] in noun_tags:
            tag_to_assign = 'NP'
        else:
            tag_to_assign = tag 
        for idx, indRange in enumerate(syntax[tag]):
            start_char = indRange[0]
            end_char = indRange[1]
            for word_indRange in list(word_pos_map.values()):
                if word_indRange[0] == start_char:
                    start_word = list(word_pos_map.keys())[list(word_pos_map.values()).index(word_indRange)]
                    break
            for word_indRange in list(word_pos_map.values()):
                if word_indRange[1] == end_char:
                    end_word = list(word_pos_map.keys())[list(word_pos_map.values()).index(word_indRange)]
                    break

            phrase_len = end_word-start_word + 1
            if phrase_len not in list(phrase_tag_map): 
                phrase_tag_map[phrase_len] = {}
                phrase_tag_map[phrase_len][tag_to_assign] = [[start_word, end_word]]
            elif tag_to_assign not in list(phrase_tag_map[phrase_len]):
                phrase_tag_map[phrase_len][tag_to_assign] = [[start_word, end_word]]
            else:
                phrase_tag_map[phrase_len][tag_to_assign].append([start_word, end_word])

    return dict(sorted(phrase_tag_map.items()))


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
    phrase_tag_map: dict - POS tag to word index mapping in the sentence. Sorted by phrase length
    word_tag_map: dict - word to index mapping in a sentence, for reverse reference of index

Output:
    results: dict - sentence to token labels mapping (each entry is a training datapoint for the model). Multiple datapoints created for one sentence from the data corpus

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
Create dataframe from datapoints
Input:
    data: dict - Mappping of sentence id to all possible datapoints (sentence to token_labels mappings) for that sentence 
    
Output:
    dataset: DataFrame - dataframe with columns ['sentence_id', 'sentence', 'tokens', 'labels']

'''
def create_dataset(data):
    dataset = []

    print('reached create dataset')
    for sentence_idx in list(data):
        for sentence in list(data[sentence_idx]):
            dataset.append({'sentence_id':sentence_idx, 'sentence':sentence, 'tokens':sentence.split(), 'labels':data[sentence_idx][sentence]})
    print('Dataframe created')
    print(pd.DataFrame(dataset))
    return pd.DataFrame(dataset)


'''
Required data-preprocessing in some cases to make the sentence tokens list and corresponding taken labels list of equal length
Input:
    dataset: DataFrame - dataframe with columns ['sentence_id', 'sentence', 'tokens', 'labels']
    
Output:
    dataset: DataFrame - Corrected dataframe to be finally used for training the model
'''
def correction(dataset):
    for index, row in dataset.iterrows():
        if len(row['tokens']) !=  len(row['labels']):
            new_tokens = row['tokens'].copy()
            new_token_ind = 0
            for ind, ch in enumerate(row['tokens']):
                if ind == len(row['tokens'])-3:
                    break
                try:
                    if ch == "'" or ch == "n'" or ch[len(ch)-1]=='-':
                        if row['tokens'][ind+1] == '|':
                            new_tokens[new_token_ind] = ch + row['tokens'][ind+2]
                            del new_tokens[ind+2]
                        else:
                            new_tokens[new_token_ind] = ch + row['tokens'][ind+1]
                            del new_tokens[ind+1]
                    elif ch == "'t" or ch == "'n" or ch[0]=='-':
                        if row['tokens'][ind-1] == '|':
                            new_tokens[new_token_ind] = ch + row['tokens'][ind-2]
                            del new_tokens[ind-2]
                        else:
                            new_tokens[new_token_ind] = ch + row['tokens'][ind-1]
                            del new_tokens[ind-1]
                    else:
                        new_token_ind += 1
                except Exception as e:
                    print(e)
                    print(new_tokens)
                    print(row['tokens'])
                    print(ind)
                    print(new_token_ind)

            dataset.loc[index].tokens = new_tokens
            dataset.loc[index].sentence = ' '.join(new_tokens)
    return dataset

if __name__ == '__main__':
    f = open('DataFiles/ontonotes5.json', errors='ignore')
    data = json.load(f)['TRAINING']
    data = list(filter(lambda d: d['language'] == 'english',data))


    # Set paths for cleaned data files
    datapoints_file_path = 'DataFiles/processed_datapoints.pkl'
    datapoints_file_path_backup = 'DataFiles/processed_datapointsbackup.pkl'
    dataset_file_path = 'DataFiles/model_dataset_alphaLables.pkl'
    filtered_dataset_file_path = 'DataFiles/model_dataset_alphaLablesFiltered.pkl'

    if os.path.exists(datapoints_file_path):
        datapoints = read_datapoints(datapoints_file_path)
        start_idx = max(list(datapoints))
    else:
        datapoints = {}
        start_idx = 0
    not_saved = 0
    end_idx = len(data)

    print(len(data))


    # Uncomment next two lines to test on specified set of examples
    #ex_list = [3]
    #for ex in ex_list:

    t_start =  time.time()
    t1 = time.time()

    #Comment next line to test on specified set of examples
    for ex in range(start_idx, end_idx):
        print(data[ex])
        
        cleaned_text = data[ex]['text']

        if ex%10000 == 0:
            store_datapoints(datapoints_file_path_backup, datapoints)
            cont = input('Do you wish to continue(y/n)?')
            if cont =='n':
                break
            print(ex,' : ','Time per 10000 datapoints ',(time.time()-t1),' secs. Total time ',(time.time()-t_start),' secs.')
            print()
            t1 = time.time()
        #print(data[ex]['text'])
        #print(cleaned_text)

        try:
            word_pos_map, word_list = create_word_pos_map(cleaned_text, data[ex]['morphology'])
            word_tag_map = create_word_tag_map(word_pos_map, data[ex]['morphology'])
            print(word_pos_map)
            print(word_tag_map)
            phrase_tag_map = create_phrase_tag_map(word_pos_map, data[ex]['syntax'] )
            print(phrase_tag_map)

            results = create_training_datapoints(word_list, phrase_tag_map, word_tag_map)
            datapoints[ex] = results
            store_datapoints(datapoints_file_path, datapoints)

        except Exception as e:
            not_saved += 1
            #print('Couldn\'t successfully create datapoints for sentence:')
            #print(data[ex]['text'])
            #print(cleaned_text)
            #print('Error: ',e)
            print('couldn\'t save count ',str(not_saved),' out of ',str(ex))
            pass

    label_ids = {'O': 0, 'N': 1 ,'R-NP': 2, 'R': 3, 'D': 4}

        
    dataset = create_dataset(datapoints, label_ids)
    dataset = correction(dataset)
    dataset.to_pickle(dataset_file_path)

    filtered_dataset = dataset[dataset.tokens.map(len) == dataset.labels.map(len)]
    filtered_dataset.to_pickle(filtered_dataset_file_path)