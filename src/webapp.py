import config, os
from train import predict

port = os.getenv("PORT", default=8000)

import streamlit as st
st.write("# English to Indic Translation")


english = st.text_input("Enter text to be translated to Indic")

language = st.selectbox("Select language to translate",["Hindi", "Tamil", "Bangla", "Kannada", "Hebrew"])

if st.button('Translate'):
    if language == "Hindi":
        translated = predict(english, 'EnHi', config.hindi)
    elif language == "Tamil":
        translated = predict(english, 'EnTa', config.tamil)
    elif language == "Bangla":
        translated = predict(english, 'EnBa', config.bangla)
    elif language == "Kannada":
        translated = predict(english, 'EnKa', config.kannada)
    else:
        translated = predict(english, 'EnHe', config.hebrew)
    
    st.subheader(translated)