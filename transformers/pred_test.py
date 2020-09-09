# -*- coding: utf-8 -*-

from transformers import TransfoXLTokenizer, TFTransfoXLLMHeadModel,GPT2LMHeadModel,GPT2Tokenizer,AutoModelForCausalLM, AutoTokenizer, XLMTokenizer,XLMWithLMHeadModel, BertForNextSentencePrediction, BertTokenizer
import torch

def prepare_xlm_input(xlm_language,model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if xlm_language in available_languages:
            language = xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text

def chatbot(md,tk,model_name,rt, testing=False):
    tokenizer = tk.from_pretrained(model_name)
    model = md.from_pretrained(model_name)
    #if model_name == "xlm-mlm-100-1280":
    #    model.config.lang_id = model.config.lang2id["no"]
    #model.to('cpu')
    while (1):
        step = 0
        text = input(">>User: ")
        if text == 'quit':
            break
        if model_name == "xlm-mlm-100-1280":
            text = prepare_xlm_input('no',model, tokenizer, text)
            print ('prepared text')

        #new_user_input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
        #print ('encoded text')
        new_user_input_ids = tokenizer.encode(text+tokenizer.eos_token,return_tensors=rt)

        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step>0 else new_user_input_ids
        chat_history_ids = model.generate(bot_input_ids,
                                        do_sample = True,
                                        max_length = 1000,
                                        pad_token_id = tokenizer.eos_token_id,
                                        top_p=0.92,
                                        top_k = 50)
        if testing:
            print (model_name + ": ",chat_history_ids.shape)
            print (model_name + ': ',bot_input_ids.shape[-1])
        print(model_name + ": {}".format(tokenizer.decode(chat_history_ids[:,bot_input_ids.shape[-1]:][0],skip_special_tokens=True)))
        step+=1

if __name__ == '__main__':
    #tk = AutoTokenizer
    tk = GPT2Tokenizer
    #tk = XLMTokenizer
    #tk = BertTokenizer
    tk = TransfoXLTokenizer
    #md = AutoModelForCausalLM
    #md = GPT2LMHeadModel
    #md = XLMWithLMHeadModel
    #md = BertForNextSentencePrediction
    md = TFTransfoXLLMHeadModel
    #chatbot(md, tk,"bert-base-multilingual-cased", testing=True)
    #chatbot(md,tk,"microsoft/DialoGPT-medium", 'pt')
    #chatbot(md,tk,"xlm-mlm-100-1280")
    chatbot(md,tk,'transfo-xl-wt103', 'tf', testing = True)

    #chatbot("xlnet-base-cased", testing=True)
