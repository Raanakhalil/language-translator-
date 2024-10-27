# Install necessary libraries
!pip install transformers streamlit

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load translation models from Hugging Face
model_name_en_ur = "Helsinki-NLP/opus-mt-en-ur"
model_name_ur_en = "Helsinki-NLP/opus-mt-ur-en"

tokenizer_en_ur = AutoTokenizer.from_pretrained(model_name_en_ur)
model_en_ur = AutoModelForSeq2SeqLM.from_pretrained(model_name_en_ur)

tokenizer_ur_en = AutoTokenizer.from_pretrained(model_name_ur_en)
model_ur_en = AutoModelForSeq2SeqLM.from_pretrained(model_name_ur_en)

# Translation function
def translate(text, direction="en-ur"):
    if direction == "en-ur":
        tokenizer = tokenizer_en_ur
        model = model_en_ur
    else:
        tokenizer = tokenizer_ur_en
        model = model_ur_en
    
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Streamlit App Interface
st.title("English-Urdu Translation Chatbot")

# Input text box for user input
input_text = st.text_area("Enter text here:")

# Radio button for selecting translation direction
lang_direction = st.radio("Select Translation Direction:", ("English to Urdu", "Urdu to English"))

# Button to translate
if st.button("Translate"):
    direction = "en-ur" if lang_direction == "English to Urdu" else "ur-en"
    translation = translate(input_text, direction)
    st.write("**Translation:**")
    st.write(translation)
