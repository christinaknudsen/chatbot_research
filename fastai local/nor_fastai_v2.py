from fastai.text import *
import pickle
import os.path
from os import path as pth

def dataload(path, datafile):
    bs=12
    itos = pickle.load( open( "data/models/norwegian_itos.pkl", "rb") )
    vocab = Vocab(itos)
    data_lm = (TextList.from_folder(path, vocab=vocab)
               #Inputs: all the text files in path
                .filter_by_folder(include=['train', 'test'])
               #We may have other temp folders that contain text files so we only keep what's in train and test
                .split_by_rand_pct(0.1)
               #We randomly split and keep 10% (10,000 reviews) for validation
                .label_for_lm()
               #We want to do a language model so we label accordingly
                .databunch(bs=bs))
    print ('loaded data')
    data_lm.show_batch()
    data_lm.save(datafile)

def training(path,datafile, model, pth_file, pkl_file):
    bs=12
    data_lm = load_data(path, datafile, bs=bs)
    config = awd_lstm_lm_config.copy()
    config['n_hid'] = 1150
    #learn = language_model_learner(data_lm,AWD_LSTM, config=config, pretrained = False, pretrained_fnames=[pth_file, pkl_file], drop_mult=0.3)
    #learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
    #data_lm = TextLMDataBunch.from_df(path=“res/model/ulmfit”, train_df=df_train, valid_df=df_valid, vocab=vocab)
    learn = language_model_learner(data_lm, AWD_LSTM, config=config,pretrained=False, drop_mult=0.5)
    learn.load_encoder('norwegian_enc')
    learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
    learn.unfreeze()
    learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
    learn.save(model)

def load_learner(path, datafile, model):
    bs=12
    #data_lm = load_data(path, datafile, bs=bs)
    #learn = language_model_learner(data_lm,AWD_LSTM, pretrained = False, pretrained_fnames=[pth_file, pkl_file], drop_mult=0.3)
    #learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
    data_lm = load_data(path, datafile, bs=bs)
    config = awd_lstm_lm_config.copy()
    config['n_hid'] = 1150
    #learn = language_model_learner(data_lm,AWD_LSTM, config=config, pretrained = False, pretrained_fnames=[pth_file, pkl_file], drop_mult=0.3)
    #learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
    #data_lm = TextLMDataBunch.from_df(path=“res/model/ulmfit”, train_df=df_train, valid_df=df_valid, vocab=vocab)
    learn = language_model_learner(data_lm, AWD_LSTM, config=config,pretrained=False, drop_mult=0.5)
    learn.load(model);

    N_WORDS = 20
    N_SENTENCES = 1
    while (1):
        try:
            sentence = input('Enter text: ')
            sentence = str(sentence)
            if sentence == ('quit'): break
            print("\n".join(learn.predict(sentence, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
        except KeyError:
            print ('error')
def run():
    path = "data"
    if pth.exists('data/data_nor_lm_v3.pkl'):
        print ('Found datafile')
        if pth.exists('data/models/fine_tuned_nor_v3.pth'):
            print ('Loading model')
            load_learner(path=path, datafile='data_nor_lm_v3.pkl', model='fine_tuned_nor_v3')

        else:
            print ('Training')
            training(path=path, datafile='data_nor_lm_v3.pkl', model='fine_tuned_nor_v3',
                    pth_file='norwegian_enc', pkl_file='norwegian_itos')

    else:
        print ('Creating datafile')
        dataload(path, 'data_nor_lm_v3.pkl')
        print ('Training')
        training(path=path, datafile='data_nor_lm_v3.pkl', model='fine_tuned_nor_v3',
                pth_file='norwegian_enc', pkl_file='norwegian_itos')
        load_learner(path=path, datafile='data_nor_lm_v3.pkl', model='fine_tuned_nor_v3')


if __name__ == '__main__':

    run()
