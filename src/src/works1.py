import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

init_data = "../data/init_data_2.txt"
with open(init_data, "r") as file:
    target_lines = [line.replace("\n", "").strip() for line in file.readlines()]

# Init tokenizer
tokenizer = Tokenizer()
# tokenizer.fit_on_texts(input_texts + target_texts)
init_texts = [
    "Ciao, come stai?",
    "Posso aiutarti con qualcosa?",
    "Qual è il tuo film preferito?",
]
tokenizer.fit_on_texts(target_lines)

model = load_model("chatbot_model.keras")

input_text = "Ciao, come stai?"

max_sequence_length = 200
input_sequence = tokenizer.texts_to_sequences([input_text])
input_sequence = pad_sequences(
    input_sequence, maxlen=max_sequence_length, padding="post"
)
# Generate answer
output_sequence = model.predict(input_sequence)

# print(predicted_index)
# print(predicted_index[0])
# print(tokenizer.index_word)

# "Decode" answer
predicted_index = np.argmax(output_sequence[0], axis=-1)
predicted_word_index = (
    predicted_index[0] if isinstance(predicted_index, np.ndarray) else predicted_index
)
# Get answer
predicted_text = tokenizer.index_word.get(predicted_word_index, "Parola sconosciuta")

print(predicted_text)
