import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.models import load_model

def create_enocder_decoder_model(lang1_vocab_size, lang2_vocab_size):
    
    hidden_dim = 256
    
    # Encoder model
    encoder_inputs = Input(shape=(None, lang1_vocab_size))
    encoder = GRU(hidden_dim, return_state=True)
    encoder_outputs, state_h = encoder(encoder_inputs)
    
    # Decoder model
    
    decoder_inputs = Input(shape=(None, lang2_vocab_size))
    decoder_gru = GRU(hidden_dim, return_sequences=True, return_state=True)
    decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=state_h)
    decoder_dense = Dense(lang2_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    return model

def get_inference_model(model):
    hidden_dim = 256
    # inference
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc = model.layers[2].output
    encoder_model = Model(encoder_inputs, state_h_enc)
    
    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(hidden_dim,), name="input_3")
    
    decoder_gru = model.layers[3]
    decoder_outputs, state_h_dec = decoder_gru(decoder_inputs, initial_state=decoder_state_input_h)
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs, decoder_state_input_h] , [decoder_outputs, state_h_dec])
    
    return encoder_model, decoder_model

def decode_sequence(input_seq, model, dic):
    
    encoder_model, decoder_model = get_inference_model(model)
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, dic['vocab_size']))
    target_seq[0, 0, dic['char_index']["\t"]] = 1.0

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h = decoder_model.predict([target_seq, states_value])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = dic['reverse_char_index'][sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == "\n" or len(decoded_sentence) > 20:
            stop_condition = True

        target_seq = np.zeros((1, 1, dic['vocab_size']))
        target_seq[0, 0, sampled_token_index] = 1.0

        states_value = h
    return decoded_sentence

def predict(input_text, language, input_dic, target_dic):
    input_seq = np.zeros((1, input_dic['max_length'], input_dic['vocab_size']), dtype="float32")
    for t, char in enumerate(input_text.upper()):
        input_seq[0, t, input_dic['char_index'][char]] = 1.0
    input_seq[0, t + 1 :, input_dic['char_index'][" "]] = 1.0
    model = load_model(f"./models/gru/{language}.h5")
    translation = decode_sequence(input_seq, model, target_dic)
    return translation