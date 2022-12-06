from dataset import Eng_Indic_Eng_Dataset
import config

import numpy as np
from pathlib import Path
import lstm, gru, lstm_attention
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import EarlyStopping

def train(dataset, model, model_name, epochs, language, save_as_pb = False):
    # es = EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=5)
    encoder_input_data, decoder_input_data, decoder_target_data = dataset.encode_data()
    model.compile(
        optimizer=Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        epochs=epochs,
        batch_size=64,
        validation_split=0.2,
    )
    if save_as_pb:
        model.save(f"./models/{model_name}/{language}")
    else:
        model.save(f"./models/{model_name}/{language}.h5")


if __name__ == "__main__":

    ## LSTM Attention

    lang = "TaEn"
    data_xml = Path(f"./input/Training/NEWS2012-Training-EnTa.xml")
    input_dic, output_dic = config.prepare_input_output_dic(
        config.tamil_alphabets, config.english_alphabets
    )

    dataset = Eng_Indic_Eng_Dataset(
        data_xml, input_dic["alphabets"], output_dic["alphabets"]
    )
    encoder_input_data, decoder_input_data, decoder_target_data = dataset.encode_data()
    model = lstm_attention.create_enocder_decoder_model(
        input_dic['vocab_size'], output_dic['vocab_size'], dataset.max_lang2_length
    )
    model_name = "lstm_attention"
    train(dataset, model, model_name, 25, lang, True)
    model = load_model(f"./models/{model_name}/{lang}")
    i = np.random.choice(len(dataset.lang1))
    input_seq = encoder_input_data[i : i + 1]
    translation = lstm_attention.decode_sequence(input_seq, model, output_dic)
    print("-")
    print("Input:", dataset.lang1[i])
    print("Orginal:", dataset.lang2[i])
    print("Translation:", translation)

    # input_dic, output_dic = config.prepare_input_output_dic(
    #     config.tamil_alphabets, config.english_alphabets
    # )
    # print(lstm_attention.predict("Malla", "EnTa", input_dic, output_dic))
