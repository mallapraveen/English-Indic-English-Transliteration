import os

port = os.getenv("PORT", default=8000)

import config
import lstm, gru
import streamlit as st

mapping = {
    "Hindi": (
        "EnHi",
        config.prepare_input_output_dic(
            config.english_alphabets, config.hindi_alphabets
        ),
        "HiEn",
        config.prepare_input_output_dic(
            config.hindi_alphabets, config.english_alphabets
        ),
    ),
    "Tamil": (
        "EnTa",
        config.prepare_input_output_dic(
            config.english_alphabets, config.tamil_alphabets
        ),
        "TaEn",
        config.prepare_input_output_dic(
            config.tamil_alphabets, config.english_alphabets
        ),
    ),
    "Bangla": (
        "EnBa",
        config.prepare_input_output_dic(
            config.english_alphabets, config.bangla_alphabets
        ),
        "BaEn",
        config.prepare_input_output_dic(
            config.bangla_alphabets, config.english_alphabets
        ),
    ),
    "Kannada": (
        "EnKa",
        config.prepare_input_output_dic(
            config.english_alphabets, config.kannada_alphabets
        ),
        "KaEn",
        config.prepare_input_output_dic(
            config.kannada_alphabets, config.english_alphabets
        ),
    ),
    "Hebrew": (
        "EnHe",
        config.prepare_input_output_dic(
            config.english_alphabets, config.hebrew_alphabets
        ),
        "HeEn",
        config.prepare_input_output_dic(
            config.hebrew_alphabets, config.hebrew_alphabets
        ),
    ),
}

st.title("English to Indic translation using seq2seq models(LSTM, GRU, LSTM with Attention, Transformer-based)")

st.write("# English to Indic Translation")
arch = st.selectbox(
    "Select architecture to translate", ["LSTM", "GRU", "LSTM_BahdanauAttention","Transformer"], key=1
)
english = st.text_input("Enter text to be translated to Indic")
language = st.selectbox(
    "Select language to translate", ["Hindi", "Tamil", "Bangla", "Kannada"]
)
if st.button("English to Indic"):
    if arch == "LSTM":
        name, input_dic, output_dic = (
            mapping[language][0],
            mapping[language][1][0],
            mapping[language][1][1],
        )
        translated = lstm.predict(english, name, input_dic, output_dic)
        st.subheader(translated)
    elif arch == "GRU":
        name, input_dic, output_dic = (
            mapping[language][0],
            mapping[language][1][0],
            mapping[language][1][1],
        )
        translated = gru.predict(english, name, input_dic, output_dic)
        st.subheader(translated)
    else:
        pass


st.write("# Indic to English Translation")
arch = st.selectbox(
    "Select architecture to translate", ["LSTM", "GRU", "LSTM_BahdanauAttention", "Transformer"], key=2
)
language = st.selectbox(
    "Select Indic language to translate",
    ["Hindi", "Tamil", "Bangla", "Kannada"],
)
indic = st.text_input(f"Enter {language} text to be translated to English")
if st.button("Indic to English"):
    if arch == "LSTM":
        name, input_dic, output_dic = (
            mapping[language][2],
            mapping[language][3][0],
            mapping[language][3][1],
        )
        translated = lstm.predict(indic, name, input_dic, output_dic)
        st.subheader(translated)
    elif arch == "GRU":
        name, input_dic, output_dic = (
            mapping[language][2],
            mapping[language][3][0],
            mapping[language][3][1],
        )
        translated = gru.predict(indic, name, input_dic, output_dic)
        st.subheader(translated)
    else:
        pass


st.subheader("For source code please visit my git [link](https://github.com/mallapraveen/Document-Extractor)")
st.subheader("[My Profile](https://github.com/mallapraveen)")
