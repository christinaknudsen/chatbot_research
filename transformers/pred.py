# -*- coding: utf-8 -*-

'''
import tensorflow as tf
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# add the EOS token as PAD token to avoid warnings
model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-base", pad_token_id=tokenizer.eos_token_id)

# encode context the generation is conditioned on
input_ids = tokenizer.encode('Jeg liker katter', return_tensors='tf')

# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# deactivate top_k sampling and sample only from 92% most likely words
sample_output = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    top_p=0.92,
    top_k=0
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
'''
'''
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

inputs = tokenizer.encode("Hello, my dog is cute", return_tensors="tf")
outputs = model.generate(inputs, do_sample=True,
max_length=50,
top_p=0.92,
top_k=0)
logits = outputs[0]
print (tokenizer.decode(logits))
from transformers import XLMTokenizer, TFXLMWithLMHeadModel
import tensorflow as tf

tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-xnli15-1024")
model = TFXLMWithLMHeadModel.from_pretrained("xlm-mlm-xnli15-1024")

inputs = tokenizer.encode("Jeg liker hunden min", return_tensors="tf")
outputs = model.generate(inputs,
                        do_sample=True,
                        max_length=50,
                        top_p=0.92,
                        top_k=0)
logits = outputs[0]
print (tokenizer.decode(logits))
'''
from transformers import AutoModelForCausalLM, AutoTokenizer, XLMTokenizer,XLMWithLMHeadModel, BertForNextSentencePrediction, BertTokenizer
import torch

def chatbot(md,tk,model_name, testing=False):
    tokenizer = tk.from_pretrained(model_name)
    model = md.from_pretrained(model_name)
    if model_name == "xlm-mlm-100-1280":
        model.config.lang_id = model.config.lang2id["no"]
    #model.to('cpu')
    while (1):
        step = 0
        text = input(">>User: ")
        if text == 'quit':
            break
        new_user_input_ids = tokenizer.encode(text+tokenizer.eos_token,return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step>0 else new_user_input_ids
        chat_history_ids = model.generate(bot_input_ids, max_length = 1000, pad_token_id = tokenizer.eos_token_id)
        if testing:
            print (model_name + ": ",chat_history_ids.shape)
            print (model_name + ': ',bot_input_ids.shape[-1])
        print(model_name + ": {}".format(tokenizer.decode(chat_history_ids[:,bot_input_ids.shape[-1]:][0],skip_special_tokens=True)))
        step+=1

if __name__ == '__main__':
    tk = AutoTokenizer
    #tk = XLMTokenizer
    #tk = BertTokenizer
    md = AutoModelForCausalLM
    #md = XLMWithLMHeadModel
    #md = BertForNextSentencePrediction
    #chatbot(md, tk,"bert-base-multilingual-cased", testing=True)
    chatbot(md,tk,"microsoft/DialoGPT-medium")

    #chatbot("xlnet-base-cased", testing=True)
