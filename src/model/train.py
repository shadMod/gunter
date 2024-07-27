import json
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from model.preprocess import preprocess_text


# Caricamento dei dati di allenamento esistenti
def load_training_data():
    if os.path.exists("training_data.json"):
        with open("training_data.json", "r") as file:
            data = json.load(file)
        return data["sentences"], data["labels"]
    else:
        return [], []


training_sentences, training_labels = load_training_data()

# Frasi di addestramento iniziali
initial_sentences = [
    "What's the weather like in Paris?",
    "Tell me the weather in New York",
    "How's the weather in London?",
    "Weather in Tokyo",
    "Temperature in Berlin",
    "Hi",
    "Hello",
    "How are you?",
    "What is your name?",
    "Bye",
    "Goodbye",
    "What should I do in the mountains?",
    "Recommend me a mountain trail",
    "Suggest a hiking route",
    "Best mountain for climbing",
    "Mountain trails near me",
    "Weather forecast for Madrid",
    "Is it raining in Rome?",
    "Snow conditions in Zurich",
    "Wind speed in Amsterdam",
    "Humidity in Dubai",
    "Weather in Barcelona",
    "Weather in Seoul",
    "How's the weather in Mumbai?",
    "Weather in Sydney",
    "Temperature in Moscow",
    "Hi there",
    "Greetings",
    "How's it going?",
    "What's up?",
    "Hey",
    "See you",
    "Talk to you later",
    "Good night",
    "Have a great day",
    "What can I do in the mountains?",
    "Any good hiking trails?",
    "Can you recommend a mountain path?",
    "Hiking spots nearby",
    "Mountain excursions",
]

initial_labels = [
    "weather",
    "weather",
    "weather",
    "weather",
    "weather",
    "greeting",
    "greeting",
    "greeting",
    "name",
    "goodbye",
    "goodbye",
    "recommendation",
    "recommendation",
    "recommendation",
    "recommendation",
    "recommendation",
    "weather",
    "weather",
    "weather",
    "weather",
    "weather",
    "weather",
    "weather",
    "weather",
    "weather",
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "goodbye",
    "goodbye",
    "goodbye",
    "goodbye",
    "recommendation",
    "recommendation",
    "recommendation",
    "recommendation",
    "recommendation",
    "recommendation",
]

# Aggiungere frasi di addestramento iniziali ai dati esistenti
training_sentences.extend(initial_sentences)
training_labels.extend(initial_labels)

processed_sentences = [preprocess_text(sentence) for sentence in training_sentences]

# Tokenizzazione
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(processed_sentences)
sequences = tokenizer.texts_to_sequences(processed_sentences)
padded_sequences = pad_sequences(sequences, padding="post")

# Pre-elaborazione delle etichette
label_encoder = LabelEncoder()
training_labels_encoded = label_encoder.fit_transform(training_labels)
training_labels_encoded = np.array(training_labels_encoded)

# Creazione del modello
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(
            input_dim=100, output_dim=16, input_length=padded_sequences.shape[1]
        ),
        tf.keras.layers.SpatialDropout1D(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(set(training_labels)), activation="softmax"),
    ]
)

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Allenamento del modello
model.fit(
    padded_sequences,
    training_labels_encoded,
    epochs=100,
    validation_split=0.2,
    batch_size=32,
)


# Funzione per riaddestrare il modello con nuovi dati
def retrain_model():
    global model, padded_sequences, training_labels_encoded
    processed_sentences = [preprocess_text(sentence) for sentence in training_sentences]
    sequences = tokenizer.texts_to_sequences(processed_sentences)
    padded_sequences = pad_sequences(sequences, padding="post")
    training_labels_encoded = label_encoder.fit_transform(training_labels)
    training_labels_encoded = np.array(training_labels_encoded)

    model.fit(
        padded_sequences,
        training_labels_encoded,
        epochs=10,
        validation_split=0.2,
        batch_size=32,
    )


# Aggiunta di nuovi dati di allenamento
def add_training_data(new_sentence, new_label):
    training_sentences.append(new_sentence)
    training_labels.append(new_label)
    with open("training_data.json", "w") as file:
        json.dump({"sentences": training_sentences, "labels": training_labels}, file)
    retrain_model()
