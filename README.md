# English - Indic -English Transliteration

This project is focused on word to to word transliteration from English to Indic languages and vice-versa. . This was done using seq2seq architecture using **LSTM** and **GRU** and **LSTM with Bahdanau Attention mechanism**.

# Requirements
- [requirements.txt](https://github.com/mallapraveen/English-Indic-English-Transliteration/blob/main/requirements.txt)

# Table Of Contents
-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)

# In a Nutshell

1. We gathered the dataset(text) from Internet. You can go the [input](https://github.com/mallapraveen/English-Indic-English-Transliteration/tree/main/input) folder to get the dataset.
2. We have written code for seq2seq models using LSTM, GRU and LSTM with attention which can be found in [src](https://github.com/mallapraveen/English-Indic-English-Transliteration/tree/main/src) folder.
3. We then trained 8 models(4 for Eng-Indic & 4 for Indic-Eng) for each architecture here. The models can be found in [models](https://github.com/mallapraveen/English-Indic-English-Transliteration/tree/main/models) folder.
4. We have then hosted the webapp in streamlit. [Link](https://mallapraveen-english-indic-english-translitera-srcwebapp-jui6w1.streamlit.app/) for the website to try it out.

# In Details
```
├──  input
│    └── *.xml - datasets for English to Indic Languages.
│
│
├──  models  
│    └── *  - 8 models for 3 different architecture(LSTM, GRU and LSTM_attn).
│ 
│
├──  src
│    └── config.py  - configuration for different languages
│    └── dataset.py - dataset generator
│    └── language_preprocessing.py - preprocessing text
│    └── train.py - to train different models
│    └── webapp.py- streamlit webapp
│    └── gru.py- gru model
│    └── lstm.py- lstm model
│    └── lstm_attention.py- lstm_attn model


```

# Future Work

This can be extended to transformer based models as well. Currently working on it.

# Contributing

Any kind of enhancement or contribution is welcomed.