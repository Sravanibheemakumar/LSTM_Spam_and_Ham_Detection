import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# Load tokenizer
try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    st.error(f"❌ Failed to load tokenizer: {e}")
    st.stop()

# Load model
try:
    model = load_model("spam_lstm_model.keras")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Streamlit UI
st.title("📩 Spam vs Ham Detector (LSTM)")
message = st.text_area("Enter your message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Tokenize
        seq = tokenizer.texts_to_sequences([message])
        
        if not seq[0]:  # If sequence is empty (unknown words)
            st.warning("⚠️ Message contains only unknown words. Unable to classify.")
        else:
            padded = pad_sequences(seq, maxlen=100)
            pred = model.predict(padded)[0][0]

            if pred > 0.5:
                st.error("🚨 Spam Message Detected!")
            else:
                st.success("✅ It's a Ham (Not Spam) Message.")




# import streamlit as st
# import numpy as np
# from keras.models import load_model
# from keras.preprocessing.sequence import pad_sequences
# import pickle

# # Load tokenizer
# with open("tokenizer.pkl", "rb") as f:
#     tokenizer = pickle.load(f)

# # Load model
# model = load_model("spam_lstm_model.keras", compile=False)

# # Streamlit UI
# st.title("📩 Spam vs Ham Detector (LSTM)")
# message = st.text_area("Enter your message:")

# if st.button("Predict"):
#     if message.strip() == "":
#         st.warning("Please enter a message.")
#     else:
#         # Preprocess input
#         seq = tokenizer.texts_to_sequences([message])
#         padded = pad_sequences(seq, maxlen=100)

#         # Predict
#         pred = model.predict(padded)[0][0]

#         if pred > 0.5:
#             st.error("🚨 Spam Message Detected!")
#         else:
#             st.success("✅ It's a Ham (Not Spam) Message.")




 



