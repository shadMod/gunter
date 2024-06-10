import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Init data
init_data = [
    ("Ciao, come stai?", "Sto bene, grazie! E tu?"),
    ("Qual è il tuo film preferito?", "Mi piace molto 'Inception'."),
]

# Init tokenizer
tokenizer = Tokenizer()
# all_texts = [pair[0] for pair in initial_data] + [pair[1] for pair in initial_data]
# tokenizer.fit_on_texts(all_texts)
answers = [pair[0] for pair in init_data]
tokenizer.fit_on_texts(answers)

# model params
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 50
lstm_units = 64
max_sequence_length = 20  # Assicurati che sia coerente con i dati

# init model
model = Sequential(
    [
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_sequence_length,
        ),
        LSTM(lstm_units, return_sequences=True),
        Dense(vocab_size, activation="softmax"),
    ]
)

# compile module with sparse or not?
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


# training model data and save/update chatbot_model.keras
def train_with_new_data(input_text, target_text):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    target_sequence = tokenizer.texts_to_sequences([target_text])

    input_sequence = pad_sequences(
        input_sequence, maxlen=max_sequence_length, padding="post"
    )
    target_sequence = pad_sequences(
        target_sequence, maxlen=max_sequence_length, padding="post"
    )

    model.fit(input_sequence, target_sequence, epochs=1, verbose=2)
    model.save("chatbot_model.keras")


# Generate answer
def generate_response(input_text):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence = pad_sequences(
        input_sequence, maxlen=max_sequence_length, padding="post"
    )

    output_sequence = model.predict(input_sequence)
    predicted_index = np.argmax(output_sequence[0], axis=-1)

    response_tokens = [tokenizer.index_word.get(idx, "") for idx in predicted_index]
    response_text = " ".join(response_tokens).strip()
    return response_text


# main __name__
while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    bot_response = generate_response(user_input)
    print(f"Bot: {bot_response}")

    # Check the correct answer by asking the user
    user_feedback = input("Was the response correct? (yes/no): ")

    if user_feedback.lower() == "no":
        correct_response = input("What should the correct response be? ")
        train_with_new_data(user_input, correct_response)
        print("The model has been updated with the new data.")

print("Chatbot session end.")
