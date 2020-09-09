Program for å teste transformers og forhåndstrente modeller

chat_bert_no.py laster inn checkpoints, vocab og config, og starter dialog.

convert_old_models.py konverterer modellen til PyTorch

For å kjøre filene må den norske og den svenske modellen lastes ned og pakkes ut i henholdsvis norwegian_bert_uncased og swedish_bert_uncased.

Modellen er nødt til å konverteres for å få riktig format til transformer sin BertLMHeadModel.
Derfor må convert_old_models.py kjøres først.

Foreløpig fungerer dette kun for den svenske modellen. Jeg vet ikke hva årsaken er, annet enn at den norske vocab-filen ser ufullstendig ut (har kun ord med ##).
