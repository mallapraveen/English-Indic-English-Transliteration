from dataset import Eng_Indic_Eng_Dataset
import config

import numpy as np
from pathlib import Path
import lstm, gru
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


def train(dataset, model, model_name, epochs, language):
    encoder_input_data, decoder_input_data, decoder_target_data = dataset.encode_data()
    model.compile(optimizer=Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, epochs=epochs, batch_size=64, validation_split=0.2)
    model.save(f"./models/{model_name}/{language}.h5")

if __name__ == "__main__":
    
    
    ## LSTM
    
    
    # lang = 'KaEn'
    # data_xml = Path(f'./input/Training/NEWS2012-Training-EnKa.xml')
    # input_dic, output_dic = config.prepare_input_output_dic(config.kannada_alphabets, config.english_alphabets)
    
    # dataset = Eng_Indic_Eng_Dataset(data_xml, input_dic['alphabets'], output_dic['alphabets'])
    # encoder_input_data, decoder_input_data, decoder_target_data = dataset.encode_data()
    # model = lstm.create_enocder_decoder_model(dataset.lang1_vocab_size, dataset.lang2_vocab_size)
    # model_name = 'lstm'
    # #train(dataset, model, model_name, 75, lang)
    # model = load_model(f"./models/{model_name}/{lang}.h5")
    # i = np.random.choice(len(dataset.lang1))
    # input_seq = encoder_input_data[i:i+1]
    # translation = lstm.decode_sequence(input_seq, model, output_dic)
    # print('-')
    # print('Input:', dataset.lang1[i])
    # print('Orginal:', dataset.lang2[i])
    # print('Translation:', translation)
    
    # input_dic, output_dic = config.prepare_input_output_dic(config.kannada_alphabets, config.english_alphabets)
    # # print(lstm.predict('Hello', 'KaEn', input_dic, output_dic))
    
    
    # lang = 'HeEn'
    # data_xml = Path(f'./input/Training/NEWS2012-Training-EnHe.xml')
    # input_dic, output_dic = config.prepare_input_output_dic(config.hebrew_alphabets, config.english_alphabets)
    
    # dataset = Eng_Indic_Eng_Dataset(data_xml, input_dic['alphabets'], output_dic['alphabets'])
    # encoder_input_data, decoder_input_data, decoder_target_data = dataset.encode_data()
    # model = lstm.create_enocder_decoder_model(dataset.lang1_vocab_size, dataset.lang2_vocab_size)
    # model_name = 'lstm'
    # train(dataset, model, model_name, 75, lang)
    # model = load_model(f"./models/{model_name}/{lang}.h5")
    # i = np.random.choice(len(dataset.lang1))
    # input_seq = encoder_input_data[i:i+1]
    # translation = lstm.decode_sequence(input_seq, model, output_dic)
    # print('-')
    # print('Input:', dataset.lang1[i])
    # print('Orginal:', dataset.lang2[i])
    # print('Translation:', translation)
    
    # input_dic, output_dic = config.prepare_input_output_dic(config.hebrew_alphabets, config.english_alphabets)
    # print(lstm.predict('Hello', 'HeEn', input_dic, output_dic))