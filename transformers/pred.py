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
'''
from transformers import XLMTokenizer, TFXLMWithLMHeadModel
import tensorflow as tf

tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-xnli15-1024")
model = TFXLMWithLMHeadModel.from_pretrained("xlm-mlm-xnli15-1024")

inputs = tokenizer.encode("Hei, jeg liker hunden min", return_tensors="tf")
outputs = model.generate(inputs,
                        do_sample=True,
                        max_length=50,
                        top_p=0.92,
                        top_k=0)
logits = outputs[0]
print (tokenizer.decode(logits))
