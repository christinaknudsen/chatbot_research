# -*- coding: utf-8 -*-
import tensorflow as tf
#from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from transformers import BartForConditionalGeneration,BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# add the EOS token as PAD token to avoid warnings
model = BartForConditionalGeneration.from_pretrained("bert-base-multilingual-cased", pad_token_id=tokenizer.eos_token_id)

# encode context the generation is conditioned on
input_ids = tokenizer.encode('Jeg liker å gå tur med hunden min', return_tensors='tf')

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
