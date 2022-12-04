import os
port = os.getenv("PORT", default=8000)

import config
import streamlit as st
from train import predict

mapping = {
    'Hindi' : ('EnHi', config.prepare_input_output_dic(config.english_alphabets, config.hindi_alphabets), 'HiEn', config.prepare_input_output_dic(config.hindi_alphabets, config.english_alphabets)),
    'Tamil' : ('EnTa', config.prepare_input_output_dic(config.english_alphabets, config.tamil_alphabets), 'TaEn', config.prepare_input_output_dic(config.tamil_alphabets, config.english_alphabets)),
    'Bangla' : ('EnBa', config.prepare_input_output_dic(config.english_alphabets, config.bangla_alphabets), 'BaEn', config.prepare_input_output_dic(config.bangla_alphabets, config.english_alphabets)),
    'Kannada' : ('EnKa', config.prepare_input_output_dic(config.english_alphabets, config.kannada_alphabets), 'KaEn', config.prepare_input_output_dic(config.kannada_alphabets, config.english_alphabets)),
    'Hebrew' : ('EnHe', config.prepare_input_output_dic(config.english_alphabets, config.hebrew_alphabets), 'HeEn', config.prepare_input_output_dic(config.hebrew_alphabets, config.hindi_english_alphabetsalphabets))
}


st.write("# English to Indic Translation")
english = st.text_input("Enter text to be translated to Indic")
language = st.selectbox("Select language to translate",["Hindi", "Tamil", "Bangla", "Kannada", "Hebrew"])
if st.button('English to Indic'):
    name, input_dic, output_dic = mapping[language][0], mapping[language][1][0], mapping[language][1][1]
    translated = predict(english, name, input_dic, output_dic)
    st.subheader(translated)

    
st.write("# Indic to English Translation")
language = st.selectbox("Select Indic language to translate", ["Hindi", "Tamil", "Bangla", "Kannada", "Hebrew"])    
indic = st.text_input(f"Enter {language} text to be translated to English")
if st.button('Indic to English'):
    name, input_dic, output_dic = mapping[language][2], mapping[language][3][0], mapping[language][3][1]
    translated = predict(indic, name, input_dic, output_dic)
    st.subheader(translated)