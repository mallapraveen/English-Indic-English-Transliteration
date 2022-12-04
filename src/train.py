from dataset import Eng_Indic_Eng_Dataset
import config

import numpy as np
from pathlib import Path
from model import create_enocder_decoder_model, get_inference_model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


def train(dataset, epochs, language):
    encoder_input_data, decoder_input_data, decoder_target_data = dataset.encode_data()

    model = create_enocder_decoder_model(dataset.lang1_vocab_size, dataset.lang2_vocab_size)
    
    model.compile(optimizer=Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, epochs=epochs, batch_size=64, validation_split=0.2)
 
    model.save(f"./models/{language}.h5")
 

def decode_sequence(input_seq, model, dic):
    
    encoder_model, decoder_model = get_inference_model(model)
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, dic['vocab_size']))
    target_seq[0, 0, dic['char_index']["\t"]] = 1.0

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = dic['reverse_char_index'][sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == "\n" or len(decoded_sentence) > 20:
            stop_condition = True

        target_seq = np.zeros((1, 1, dic['vocab_size']))
        target_seq[0, 0, sampled_token_index] = 1.0

        states_value = [h, c]
    return decoded_sentence

def predict(input_text, name, input_dic, target_dic):
    input_seq = np.zeros((1, input_dic['max_length'], input_dic['vocab_size']), dtype="float32")
    for t, char in enumerate(input_text.upper()):
        input_seq[0, t, input_dic['char_index'][char]] = 1.0
    input_seq[0, t + 1 :, input_dic['char_index'][" "]] = 1.0
    model = load_model(f"./models/{name}.h5")
    translation = decode_sequence(input_seq, model, target_dic)
    return translation

if __name__ == "__main__":
    
    
    # For training
    # Hindi
    
    lang = 'HiEn'
    data_xml = Path(f'./input/Training/NEWS2012-Training-EnHi.xml')
    input_dic, output_dic = config.prepare_input_output_dic(config.hindi_alphabets, config.english_alphabets)
    
    dataset = Eng_Indic_Eng_Dataset(data_xml, input_dic['alphabets'], output_dic['alphabets'])
    encoder_input_data, decoder_input_data, decoder_target_data = dataset.encode_data()
    train(dataset, 75, lang)
    model = load_model(f"./models/{lang}.h5")
    i = np.random.choice(len(dataset.lang1))
    input_seq = encoder_input_data[i:i+1]
    translation = decode_sequence(input_seq, model, output_dic)
    print('-')
    print('Input:', dataset.lang1[i])
    print('Orginal:', dataset.lang2[i])
    print('Translation:', translation)
    
    # input_dic, output_dic = config.prepare_input_output_dic(config.english_alphabets, config.hindi_alphabets)
    # print(predict('Hello', 'EnHi', input_dic, output_dic))