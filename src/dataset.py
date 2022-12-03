import re
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from language_preprocessing import clean_English_Vocab, clean_Vocab
    
class Eng_Indic_Dataset():
    def __init__(self, filename, lang_vocab_cleaner, alphabets):
        
        input_characters = set(list(' ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        target_characters = set(alphabets)
        
        self.input_characters = sorted(list(input_characters))
        self.target_characters = sorted(list(target_characters))
        
        self.lang1, self.lang2 = self.readXmlDataset(filename, lang_vocab_cleaner)
        
        self.lang1_vocab_size = len(input_characters)
        self.lang2_vocab_size = len(target_characters)
        self.max_lang1_length = max([len(txt) for txt in self.lang1])
        self.max_lang2_length = max([len(txt) for txt in self.lang2])
        
        print("English to Hindi Dataset")
        print("No.of samples:", len(self.lang1))
        print("No.of unique input tokens:", self.lang1_vocab_size)
        print("No.of unique output tokens:", self.lang2_vocab_size)
        print("Maximum seq length for inputs:", self.max_lang1_length)
        print("Maximum seq length for outputs:", self.max_lang2_length)
        
        self.input_token_index = dict([(char, i) for i, char in enumerate(self.input_characters)])
        self.target_token_index = dict([(char, i) for i, char in enumerate(self.target_characters)])
        
        self.reverse_input_char_index = dict((i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict((i, char) for char, i in self.target_token_index.items())
        
        self.encoder_input_data = np.zeros((len(self.lang1), self.max_lang1_length, self.lang1_vocab_size), dtype="float32")

        self.decoder_input_data = np.zeros((len(self.lang2), self.max_lang2_length, self.lang2_vocab_size), dtype="float32")

        self.decoder_target_data = np.zeros((len(self.lang2), self.max_lang2_length, self.lang2_vocab_size), dtype="float32")

    def encode_data(self):
        for i, (input_text, target_text) in enumerate(zip(self.lang1, self.lang2)):
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, self.input_token_index[char]] = 1.0
            self.encoder_input_data[i, t + 1 :, self.input_token_index[" "]] = 1.0
            for t, char in enumerate(target_text):
                self.decoder_input_data[i, t, self.target_token_index[char]] = 1.0
                if t > 0:
                    self.decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.0
            self.decoder_input_data[i, t + 1 :, self.target_token_index[" "]] = 1.0
            self.decoder_target_data[i, t:, self.target_token_index[" "]] = 1.0
            
        return self.encoder_input_data, self.decoder_input_data, self.decoder_target_data
    
    def readXmlDataset(self, filename:Path, lang_vocab_cleaner) -> tuple:
        transliterationCorpus = ET.parse(filename).getroot()
        lang1_words = []
        lang2_words = []
        for line in transliterationCorpus:
            wordlist1 = clean_English_Vocab(line[0].text)
            wordlist2 = lang_vocab_cleaner(line[1].text, self.target_characters)
            
            # Skip noisy data
            if len(wordlist1) != len(wordlist2):
                print('Skipping: ', line[0].text, ' - ', line[1].text)
                continue

            for word in wordlist1:
                lang1_words.append(word)
            for word in wordlist2:
                lang2_words.append("\t" + word + "\n")

        return lang1_words, lang2_words
    
class Indic_Eng_Dataset():
    def __init__(self, filename, lang_vocab_cleaner, alphabets):
        
        input_characters = set([' '] + alphabets)
        target_characters = set(['\t','\n',' '] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        
        self.input_characters = sorted(list(input_characters))
        self.target_characters = sorted(list(target_characters))
        
        self.lang1, self.lang2 = self.readXmlDataset(filename, lang_vocab_cleaner)
        
        self.lang1_vocab_size = len(input_characters)
        self.lang2_vocab_size = len(target_characters)
        self.max_lang1_length = max([len(txt) for txt in self.lang1])
        self.max_lang2_length = max([len(txt) for txt in self.lang2])
        
        print("English to Hindi Dataset")
        print("No.of samples:", len(self.lang1))
        print("No.of unique input tokens:", self.lang1_vocab_size)
        print("No.of unique output tokens:", self.lang2_vocab_size)
        print("Maximum seq length for inputs:", self.max_lang1_length)
        print("Maximum seq length for outputs:", self.max_lang2_length)
        
        self.input_token_index = dict([(char, i) for i, char in enumerate(self.input_characters)])
        self.target_token_index = dict([(char, i) for i, char in enumerate(self.target_characters)])
        
        self.reverse_input_char_index = dict((i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict((i, char) for char, i in self.target_token_index.items())
        
        self.encoder_input_data = np.zeros((len(self.lang1), self.max_lang1_length, self.lang1_vocab_size), dtype="float32")

        self.decoder_input_data = np.zeros((len(self.lang2), self.max_lang2_length, self.lang2_vocab_size), dtype="float32")

        self.decoder_target_data = np.zeros((len(self.lang2), self.max_lang2_length, self.lang2_vocab_size), dtype="float32")

    def encode_data(self):
        for i, (input_text, target_text) in enumerate(zip(self.lang1, self.lang2)):
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, self.input_token_index[char]] = 1.0
            self.encoder_input_data[i, t + 1 :, self.input_token_index[" "]] = 1.0
            for t, char in enumerate(target_text):
                self.decoder_input_data[i, t, self.target_token_index[char]] = 1.0
                if t > 0:
                    self.decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.0
            self.decoder_input_data[i, t + 1 :, self.target_token_index[" "]] = 1.0
            self.decoder_target_data[i, t:, self.target_token_index[" "]] = 1.0
            
        return self.encoder_input_data, self.decoder_input_data, self.decoder_target_data
    
    def readXmlDataset(self, filename:Path, lang_vocab_cleaner) -> tuple:
        transliterationCorpus = ET.parse(filename).getroot()
        lang1_words = []
        lang2_words = []
        for line in transliterationCorpus:
            wordlist1 = clean_English_Vocab(line[0].text)
            wordlist2 = lang_vocab_cleaner(line[1].text, self.target_characters)
            
            # Skip noisy data
            if len(wordlist1) != len(wordlist2):
                print('Skipping: ', line[0].text, ' - ', line[1].text)
                continue

            for word in wordlist1:
                lang1_words.append(word)
            for word in wordlist2:
                lang2_words.append("\t" + word + "\n")

        return lang2_words, lang1_words