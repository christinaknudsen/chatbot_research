Program to test transformers and pretrained models.

chat_bert_no.py loads checkpoints, vocab and config and starts a dialog

convert_tf_to_pt.py converts the model to PyTorch. This has to be run before starting the chat.

Remember to download and unpack the models before running the code.

After some testing, it seems like the models work fine in danish and swedish, but not in norwegian. 
