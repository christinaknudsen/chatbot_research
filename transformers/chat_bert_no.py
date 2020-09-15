# -*- coding: utf-8 -*-
from transformers import BertTokenizer, BertConfig, BertLMHeadModel
import tensorflow as tf

def fix_vocab(vocab_file, output_file):
    new_vocab = open(output_file, 'w', encoding="utf8")
    vocab = open(vocab_file, 'r', encoding="utf8")
    for word in vocab:
        word = word.replace("##","")
        word = word.replace("▁","##")
        new_vocab.write(word)
    print ('Done')
    new_vocab.close()
    vocab.close()

'''
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
config = BertConfig('bert-base-cased')
model = BertLMHeadModel.from_pretrained('bert-base-cased')
'''

def chat(folder_bert, voc, testing = False):
    tf.random.set_seed(1)
    tokenizer = BertTokenizer(vocab_file = folder_bert+voc)
    if testing:
        tokens = tokenizer.tokenize("jeg tror det skal regne")
        print(tokens)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        print(ids)
        print("Vocab size:", len(tokenizer.vocab))

    config = BertConfig.from_json_file(folder_bert + "/config.json")
    model = BertLMHeadModel.from_pretrained(folder_bert, config=config)
    while (1):
        text = input(">>User: ")
        if text == 'quit':
            break
        input_ids = tokenizer.encode(text+tokenizer.sep_token, return_tensors='pt')
        #print ('input_ids', input_ids)
        sample_output = model.generate(
            input_ids,
            pad_token_id = tokenizer.sep_token_id,
            do_sample=True,
            max_length=50,
            repetition_penalty = 1.8,
            top_k=20,
            #top_p=0.90,
            temperature = 0.7
        )
        print("Bot: {}".format(tokenizer.decode(sample_output[0])))
        print("Bot: {}".format(tokenizer.decode(sample_output[:,input_ids.shape[-1]:][0], skip_special_tokens=True)))

if __name__ == "__main__":
    #fix_vocab("./norwegian_bert_uncased_local/vocab.txt", "./output_norwegian_local/new_vocab.txt")
    language = 'norwegian'
    folder_bert = "./output_"+language+"_local"
    voc = "/new_vocab.txt"
    chat(folder_bert, voc, testing=True)
    print ('yey')
