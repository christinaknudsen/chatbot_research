from fastai.text import *

def trainer(path):
    bs=12
    data_lm = (TextList.from_folder(path)
               #Inputs: all the text files in path
                .filter_by_folder(include=['train', 'test', 'unsup'])
               #We may have other temp folders that contain text files so we only keep what's in train and test
                .split_by_rand_pct(0.1)
               #We randomly split and keep 10% (10,000 reviews) for validation
                .label_for_lm()
               #We want to do a language model so we label accordingly
                .databunch(bs=bs))
    data_lm.save('data_lm.pkl')
    data_lm = load_data(path, 'data_lm.pkl', bs=bs)
    data_lm.show_batch()
    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
    #learn.lr_find()
    #learn.recorder.plot(skip_end=15)
    learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
    learn.load('fit_head');
    learn.unfreeze()
    learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
    learn.save('fine_tuned')
    learn.load('fine_tuned');
    TEXT = "I like Interstellar because"
    N_WORDS = 40
    N_SENTENCES = 2
    print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
    learn.save_encoder('stage1_enc')

def load_learner(oath):
    bs=12
    data_lm = load_data(path, 'data_lm.pkl', bs=bs)
    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
    learn.load('fine_tuned');
    TEXT = "I love Brad Pitt because"
    N_WORDS = 40
    N_SENTENCES = 2
    while (1):
        try:
            sentence = input('Enter text: ')
            sentence = str(sentence)
            if sentence == ('quit'): break
            print("\n".join(learn.predict(sentence, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
            learn.save_encoder('stage1_enc')
        except KeyError:
            print ('error')

if __name__ == '__main__':
    path = untar_data(URLs.IMDB)
    print (path)
    #df = pd.read_csv(path/'texts.csv')
    #trainer(path)
    load_learner(path)
