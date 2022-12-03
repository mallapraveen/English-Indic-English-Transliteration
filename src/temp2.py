import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import *
from keras.utils import *
from keras.initializers import *
from keras.optimizers import * 
import numpy as np
import xml.etree.ElementTree as ET
from language_preprocessing import clean_English_Vocab, clean_Hindi_Vocab
from pathlib import Path
from tensorflow.keras.models import load_model

batch_size = 64  
epochs = 100  
latent_dim = 256  
num_samples = 10000  
# set the data_path accordingly
data_path = "./input/spa.txt" 


def readXmlDataset(filename:Path, lang_vocab_cleaner) -> tuple:
    transliterationCorpus = ET.parse(filename).getroot()
    lang1_words = []
    lang2_words = []
    for line in transliterationCorpus:
        wordlist1 = clean_English_Vocab(line[0].text)
        wordlist2 = lang_vocab_cleaner(line[1].text)
        
        # Skip noisy data
        if len(wordlist1) != len(wordlist2):
            print('Skipping: ', line[0].text, ' - ', line[1].text)
            continue

        for word in wordlist1:
            lang1_words.append(word)
        for word in wordlist2:
            lang2_words.append("\t" + word + "\n")

    return lang1_words, lang2_words

input_texts = []
target_texts = []
input_characters = set(list(' ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
target_characters = set(['\t','\n',' '] + [chr(alpha) for alpha in range(2304, 2432)])

input_texts, target_texts = readXmlDataset(Path('./input/Training/NEWS2012-Training-EnHi-13937.xml'), clean_Hindi_Vocab)
# print(input_characters)
# print(target_characters)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

# print("No.of samples:", len(input_texts))
# print("No.of unique input tokens:", num_encoder_tokens)
# print("No.of unique output tokens:", num_decoder_tokens)
# print("Maximum seq length for inputs:", max_encoder_seq_length)
# print("Maximum seq length for outputs:", max_decoder_seq_length)


input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# print(input_token_index)
# print(target_token_index)

encoder_input_data = np.zeros(
  (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)

decoder_input_data = np.zeros(
  (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

decoder_target_data = np.zeros(
  (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        print(t, char)
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0

# encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
# encoder = keras.layers.LSTM(latent_dim, return_state=True)
# encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# encoder_states = [state_h, state_c]

# decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))
# decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
# decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
# decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
# decoder_outputs = decoder_dense(decoder_outputs)

# model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# print(encoder_input_data[0, 1, :])
# print(decoder_input_data[0,1,:])
# print(decoder_target_data[0,0,:])

#model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001), loss='categorical_crossentropy', metrics=["accuracy"])

# model.fit(
#     [encoder_input_data, decoder_input_data],
#     decoder_target_data,
#     batch_size=batch_size,
#     epochs=50,
#     validation_split=0.2,
# )
# model.save("./models/E2S")

# model.load("./models/E2S")

model = load_model("./models/E2S")

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)


reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, target_token_index["\t"]] = 1.0
        
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0][0])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        states_value = [h, c]
    return decoded_sentence

i = np.random.choice(len(input_texts))
input_seq = encoder_input_data[i:i+1]
translation = decode_sequence(input_seq)
print('-')
print('Input:', input_texts[i])
print('Original:', target_texts[i])
print('Translation:', translation)