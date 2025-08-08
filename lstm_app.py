import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = load_model("spam_lstm_model.keras", compile=False)

# Streamlit UI
st.title("📩 Spam vs Ham Detector (LSTM)")
message = st.text_area("Enter your message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess input
        seq = tokenizer.texts_to_sequences([message])
        padded = pad_sequences(seq, maxlen=100)

        # Predict
        pred = model.predict(padded)[0][0]

        if pred > 0.5:
            st.error("🚨 Spam Message Detected!")
        else:
            st.success("✅ It's a Ham (Not Spam) Message.")

 

