# LSTM_Spam_and_Ham_Detection

📩 LSTM-Based Spam vs Ham Detection
This project is a deep learning-based SMS classification system that detects whether a given message is spam (unwanted promotional or fraudulent) or ham (legitimate).
It uses a Long Short-Term Memory (LSTM) neural network trained on labeled SMS data to capture sequential patterns in text messages.

🚀 Features
* LSTM Model: Utilizes deep learning for sequence-based text classification.
* Streamlit Web App: Interactive user interface for real-time predictions.

* Preprocessing Pipeline:
  - Text cleaning
  - Tokenization
  - Padding sequences
- Model Deployment: Easily deployable on Streamlit Cloud.

📂 Project Structure

├── app.py                       # Streamlit main app file

├── spam_lstm_model.keras        # Trained LSTM model

├── tokenizer.pkl                # Fitted tokenizer

├── requirements.txt             # Dependencies

├── README.md                    # Project documentation

└── data/ spam.csv               # (Optional) Dataset folder


🛠️ Technologies Used
 - Python 3.x
 - TensorFlow / Keras
 - Streamlit
 - NumPy, Pandas, Scikit-learn
 - h5py for model loading

📊 Model Workflow

1.Data Preprocessing: Cleaning and tokenizing the text.

2.Sequence Padding: Making all messages the same length.

3.LSTM Model Training: Using an embedding layer, LSTM units, and dense output layer.

4.Evaluation: Measuring accuracy, precision.

5.Deployment: Integrating the trained model into a Streamlit app for public use.

🌐 Live Demo
🔗 [Click here to try the app
](https://lstmspamandhamdetection-8rwhaeec2wlatzeascuxtd.streamlit.app/)
