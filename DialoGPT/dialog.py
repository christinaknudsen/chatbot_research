# -*- coding: utf-8 -*-

from transformers import GPT2LMHeadModel,GPT2Tokenizer, AutoConfig, AutoTokenizer, AutoModelWithLMHead
import torch


def chatbot(md,tk,model_name):
    tokenizer = tk.from_pretrained(model_name)
    model = md#.from_pretrained(model_name)
    model.to('cpu')
    while (1):
        step = 0
        text = input(">>User: ")
        if text == 'quit':
            break
        new_user_input_ids = tokenizer.encode(text+tokenizer.eos_token,return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step>0 else new_user_input_ids
        chat_history_ids = model.generate(bot_input_ids,
                                        do_sample = True,
                                        max_length = 1000,
                                        pad_token_id = tokenizer.eos_token_id,
                                        top_p=0.92,
                                        top_k = 50)
        print(model_name + ": {}".format(tokenizer.decode(chat_history_ids[:,bot_input_ids.shape[-1]:][0],skip_special_tokens=True)))
        step+=1

if __name__ == '__main__':
    tk = GPT2Tokenizer
    #md = GPT2LMHeadModel


    config = AutoConfig.from_pretrained("output/checkpoint-5000/"+ "/config.json")
    #tk = AutoTokenizer.from_pretrained("output/checkpoint-5000", cache_dir="data")
    md = AutoModelWithLMHead.from_pretrained("output/checkpoint-5000/pytorch_model.bin", config=config)


    chatbot(md,tk,"microsoft/DialoGPT-small")
