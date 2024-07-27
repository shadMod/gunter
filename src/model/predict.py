from model.train import tokenizer, padded_sequences, label_encoder, model
from model.preprocess import preprocess_text
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


def chatbot_response(user_input):
    processed_input = preprocess_text(user_input)
    sequence = tokenizer.texts_to_sequences([processed_input])
    padded_sequence = pad_sequences(
        sequence, maxlen=padded_sequences.shape[1], padding="post"
    )
    prediction = model.predict(padded_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]
