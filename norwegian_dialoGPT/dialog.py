from transformers import AutoTokenizer, AutoModelWithLMHead

def chatbot(model,tokenizer):
    step = 0
    while (1):
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
                                        top_k = 50
                                        )
        print("Chatbot: {}".format(tokenizer.decode(chat_history_ids[:,bot_input_ids.shape[-1]:][0],skip_special_tokens=True)))
        step+=1

if __name__ == '__main__':
    tk = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
    md = AutoModelWithLMHead.from_pretrained('output')
    chatbot(md,tk)
