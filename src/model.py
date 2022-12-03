from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

def create_enocder_decoder_model(lang1_vocab_size, lang2_vocab_size):
    
    hidden_dim = 256
    
    # Encoder model
    encoder_inputs = Input(shape=(None, lang1_vocab_size))
    encoder = LSTM(hidden_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    
    # Decoder model
    encoder_states = [state_h, state_c]
    
    decoder_inputs = Input(shape=(None, lang2_vocab_size))
    decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(lang2_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    return model

def get_inference_model(model):
    hidden_dim = 256
    # inference
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)
    
    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(hidden_dim,), name="input_3")
    decoder_state_input_c = Input(shape=(hidden_dim,), name="input_4")
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    return encoder_model, decoder_model