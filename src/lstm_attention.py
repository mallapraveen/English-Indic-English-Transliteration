import config
from train import train
import numpy as np
from pathlib import Path
from dataset import Eng_Indic_Eng_Dataset

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Layer
from keras.layers import Lambda
import keras.backend as K

class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units=units)
        self.W2 = Dense(units=units)
        self.V = Dense(units=1)
    
    # query - (1,hidden)
    # values - (batch, 1, hidden_size of input lstm)
    
    def call(self, query, values):
        # (1, 1, 256)
        query_with_time_axis = tf.expand_dims(query, 1)
        # print(query_with_time_axis.shape)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        # print(score.shape)
        # (1, 20, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        # print(attention_weights.shape)
        # shape=(1, 20, 256)
        context_vector = attention_weights * values
        # print(context_vector.shape)
        # (1, 256)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # print(context_vector.shape)
        return context_vector, attention_weights
    
# encoder_outputs = np.random.rand(1, 20, 256)
# decoder_outputs = np.random.rand(1, 256)
# attention = BahdanauAttention(256)
# attention(decoder_outputs, encoder_outputs)

def create_enocder_decoder_model(lang1_vocab_size, lang2_vocab_size, timesteps):
    hidden_dim = 256
    
    #Encoder Model
    encoder_inputs = Input(shape=(None, lang1_vocab_size))
    encoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
    encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
    encoder_states = [encoder_state_h, encoder_state_c]
    attention = BahdanauAttention(hidden_dim)
    # Decoder Model
    
    decoder_inputs = Input(shape=(None, lang2_vocab_size))
    decoder_lstm = LSTM(hidden_dim, return_state=True, recurrent_initializer="glorot_uniform")
    decoder_dense = Dense(lang2_vocab_size, activation="softmax", name="dense_output")
    
    states = encoder_states
    decoder_outputs = encoder_state_h
    # (None, 256)
    # print(decoder_outputs.shape)
    inputs = decoder_inputs[:,0,:]
    # (None, 131)
    #print(inputs.shape)
    all_outputs = []
    for i in range(timesteps):
        # context - shape=(batch, 256)  atten - (1, 20, 1)
        context_vector, attention_weights = attention(decoder_outputs, encoder_outputs)
        context_vector = tf.expand_dims(context_vector, 1)
        # (None, 1, 256)
        #print(context_vector)
        inputs = tf.expand_dims(inputs, 1)
        # (None, 1, 131)
        #print(inputs)
        inputs = tf.concat([context_vector, inputs], axis=-1)
        # (None, 1, 387)
        #print(inputs)
        decoder_outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)
        #(None, 256) (None, 256) (None, 256)
        #print(decoder_outputs.shape, state_h.shape, state_c.shape)
        outputs = decoder_dense(decoder_outputs)
        #(None, 131)
        #print(outputs.shape)
        outputs = tf.expand_dims(outputs, 1)
        # (None, 1, 131)
        #print(outputs.shape)
        all_outputs.append(outputs)
        #print(all_outputs)
        inputs = decoder_inputs[:,i,:]
        states = [state_h, state_c]
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    #print(decoder_outputs.shape)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    return model

def get_inference_model(model, lang2_vocab_size):
    hidden_dim = 256
    
    # encoder model
    encoder_inputs = model.input[0]
    encoder_lstm = model.layers[1]
    encoder_outputs, state_h_enc, state_c_enc = encoder_lstm(encoder_inputs)
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)
    # decoder model
    #decoder_inputs = model.input[1]
    decoder_inputs = Input(shape=(1,lang2_vocab_size), name="input_6")
    #print(decoder_inputs)
    decoder_state_input_h = Input(shape=(hidden_dim,), name="input_3")
    decoder_state_input_c = Input(shape=(hidden_dim,), name="input_4")
    encoder_outputs_inputs = Input(shape=(None,hidden_dim), name="input_5")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[8]
    decoder_outputs = state_h_enc
    inputs = decoder_inputs
    #print("inputs slice", inputs.shape)
    attention = model.layers[3]
    #print(decoder_state_input_h.shape, encoder_outputs_inputs.shape)
    context_vector, attention_weights = attention(decoder_state_input_h, encoder_outputs_inputs)
    #print('context shape', context_vector.shape)
    context_vector = tf.expand_dims(context_vector, 1)
    #print('context shape', context_vector.shape)
    #inputs = tf.expand_dims(inputs, 1)
    #print('inputs shape', inputs.shape)
    inputs = tf.concat([context_vector, inputs], axis=-1)
    #print('inputs shape', inputs.shape)
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.get_layer('dense_output')
    outputs = decoder_dense(decoder_outputs)
    outputs = tf.expand_dims(outputs, 1)
    decoder_model = Model([decoder_inputs, encoder_outputs_inputs] + decoder_states_inputs, [outputs] + decoder_states)
    
    return encoder_model, decoder_model
    
def decode_sequence(input_seq, model, dic):
    encoder_model, decoder_model = get_inference_model(model, dic['vocab_size'])
    encoder_outputs, state_h, state_c = encoder_model.predict(input_seq)
    states_value = [state_h, state_c]
    # print(encoder_outputs.shape, state_h.shape, state_c.shape)
    
    target_seq = np.zeros((1, 1, dic["vocab_size"]))
    target_seq[0, 0, dic["char_index"]["\t"]] = 1
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq, encoder_outputs] + states_value)
        sampled_token_index = np.argmax(output_tokens[0,-1,:])
        sampled_char = dic["reverse_char_index"][sampled_token_index]
        decoded_sentence += sampled_char
        
        if sampled_char == "\n" or len(decoded_sentence) > 20:
            stop_condition = True
            
        target_seq = np.zeros((1, 1, dic["vocab_size"]))
        target_seq[0, 0, sampled_token_index] = 1.0

        states_value = [h, c]
    return decoded_sentence

def predict(input_text, language, input_dic, target_dic):
    input_seq = np.zeros(
        (1, input_dic["max_length"], input_dic["vocab_size"]), dtype="float32"
    )
    for t, char in enumerate(input_text.upper()):
        input_seq[0, t, input_dic["char_index"][char]] = 1.0
    input_seq[0, t + 1 :, input_dic["char_index"][" "]] = 1.0
    model = load_model(f"./models/lstm_attention/{language}")
    translation = decode_sequence(input_seq, model, target_dic)
    return translation


# lang = "EnHi"
# data_xml = Path(f"./input/Training/NEWS2012-Training-EnHi.xml")
# input_dic, output_dic = config.prepare_input_output_dic(
#     config.english_alphabets, config.hindi_alphabets
# )

# dataset = Eng_Indic_Eng_Dataset(
#     data_xml, input_dic["alphabets"], output_dic["alphabets"]
# )
# encoder_input_data, decoder_input_data, decoder_target_data = dataset.encode_data()
# #model = create_enocder_decoder_model(input_dic['vocab_size'], output_dic['vocab_size'], dataset.max_lang2_length)
# #train(dataset, model, 'lstm_attention', 20, lang, True)
# model = load_model(f"./models/lstm_attention/{lang}")
# i = np.random.choice(len(dataset.lang1))
# input_seq = encoder_input_data[i : i + 1]
# translation = decode_sequence(input_seq, model, output_dic)
# print("-")
# print("Input:", dataset.lang1[i])
# print("Orginal:", dataset.lang2[i])
# print("Translation:", translation)