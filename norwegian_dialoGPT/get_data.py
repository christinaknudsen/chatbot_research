"""
Function to create a csv file with contextualized sentences. Each row in the new file will have a sentence with
current response and the 7 previous responses. Outputs are pandas dataframes for training and evaluation.
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from parameters import Args
args = Args()

def createContext(filename):
    all_lines = pd.read_csv(args.filmtekst_path+'/'+filename, delimiter = "\n", header = None, names = ['lines'])
    contexted = []
    for i in range(args.n, len(all_lines['lines'])):
        row = []
        prev = i - 1 - n # we additionally subtract 1, so row will contain current response and 7 previous responses
        for j in range(i, prev, -1):
            row.append(all_lines['lines'][j])
        contexted.append(row)
    print ('len contexted',len(contexted))
    return contexted

def createCorpus():
    if args.load_corpus:
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
