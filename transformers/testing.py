
import numpy as np
import torch

import argparse
import logging

from transformers import XLMTokenizer, XLMWithLMHeadModel
def prepare_xlm_input(xlm_language, model, tokenizer, prompt_text):
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
    return prompt_text
def main():
    length = 200
    xlm_language='no'
    temperature = 0.7
    k = 0
    p=0.9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-100-1280")
    model = XLMWithLMHeadModel.from_pretrained("xlm-mlm-100-1280")
    model.to(device)
    prompt_text = input("Model prompt >>> ")
    preprocessed_prompt_text = prepare_xlm_input(xlm_language, model, tokenizer, prompt_text)
    tokenizer_kwargs = {}
    encoded_prompt = tokenizer.encode(preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs)
    encoded_prompt = encoded_prompt.to(device)
    input_ids = encoded_prompt

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=length + len(encoded_prompt[0]),
        temperature=temperature,
        top_k=k,
        top_p=p,
        repetition_penalty=1,
        do_sample=True,
        num_return_sequences=1,
    )
    if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()
    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        #text = text[: text.find(args.stop_token) if args.stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
        print(total_sequence)

    return generated_sequences
if __name__ == "__main__":
    main()
