"""
Function to create a csv file with contextualized sentences. Each row in the new file will have a sentence with
current response and the 7 previous responses. Outputs are pandas dataframes for training and evaluation.
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from parameters import Args
import numpy as np

args = Args()

def createContext(filename):
    all_lines = pd.read_csv(args.filmtekst_path+'/'+filename, delimiter = "\n", header = None, names = ['lines'])
    contexted = []
    for i in range(args.n, len(all_lines['lines'])):
        row = []
        prev = i - 1 - args.n # we additionally subtract 1, so row will contain current response and 7 previous responses
        for j in range(i, prev, -1):
            row.append(all_lines['lines'][j])
        contexted.append(row)
    return contexted

def createCorpus(chat = True):
    if args.load_corpus and chat:
        print ('Loading corpus')
        print ('..............')
        df = pd.read_csv(args.corpus_name)
    else:
        print ('Creating corpus')
        print ('..............')
        all_contexted = []
        for idx, filename in enumerate(os.listdir(args.filmtekst_path)):
            print (filename)
            contexted = createContext(filename)
            all_contexted+=contexted
        columns = ['response', 'context']
        columns = columns + ['context/'+str(i) for i in range(args.n-1)]
        df = pd.DataFrame.from_records(all_contexted, columns=columns)
        df.to_csv(args.corpus_name, encoding = 'utf-8')

    df = df.dropna()
    df = df.drop(df.columns[0], axis=1)
    trn_df, val_df = train_test_split(df, test_size = 0.2)
    return trn_df, val_df

def split_dataset(filename, nb_sets):
    df = pd.read_csv(filename)
    df_list = np.array_split(df, nb_sets)
    for i in range (len(df_list)):
        print ('Creating set {} of {}'.format(i, len(df_list)))
        dataf = df_list[i]
        dataf = dataf.dropna()
        dataf = dataf.drop(dataf.columns[0], axis=1)
        print (dataf.head(5))
        dataf.to_csv('corpus_{}.csv'.format(i), encoding = 'utf-8')

if __name__ == "__main__":
    createCorpus(chat = False)
    #split_dataset('final_corpus_large.csv', 12)
