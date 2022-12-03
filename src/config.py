# English

english_vocab_size = 27
english_max_length = 20
english_characters = sorted(list(' ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
english_token_index = dict([(char, i) for i, char in enumerate(english_characters)])

# hindi

hindi_alphabets = sorted(['\t','\n',' '] + [chr(alpha) for alpha in range(2304, 2432)])
hindi_char_index = dict([(char, i) for i, char in enumerate(hindi_alphabets)])
hindi_reverse_char_index = dict((i, char) for char, i in hindi_char_index.items())

hindi = {
    'vocab_size' : 131,
    'alphabets' : hindi_alphabets,
    'char_index' : hindi_char_index,
    'reverse_char_index' : hindi_reverse_char_index
}

# Bangla

bangla_alphabets = sorted(['\t','\n',' '] + [chr(alpha) for alpha in range(2432, 2559)])
bangla_char_index = dict([(char, i) for i, char in enumerate(bangla_alphabets)])
bangla_reverse_char_index = dict((i, char) for char, i in bangla_char_index.items())

bangla = {
    'vocab_size' : 130,
    'alphabets' : bangla_alphabets,
    'char_index' : bangla_char_index,
    'reverse_char_index' : bangla_reverse_char_index
}

# Tamil

tamil_alphabets = sorted(['\t','\n',' '] + [chr(alpha) for alpha in range(2944, 3071)])
tamil_char_index = dict([(char, i) for i, char in enumerate(tamil_alphabets)])
tamil_reverse_char_index = dict((i, char) for char, i in tamil_char_index.items())

tamil = {
    'vocab_size' : 130,
    'alphabets' : tamil_alphabets,
    'char_index' : tamil_char_index,
    'reverse_char_index' : tamil_reverse_char_index
}

# Kannada

kannada_alphabets = sorted(['\t','\n',' '] + [chr(alpha) for alpha in range(3200, 3327)])
kannada_char_index = dict([(char, i) for i, char in enumerate(kannada_alphabets)])
kannada_reverse_char_index = dict((i, char) for char, i in kannada_char_index.items())

kannada = {
    'vocab_size' : 130,
    'alphabets' : kannada_alphabets,
    'char_index' : kannada_char_index,
    'reverse_char_index' : kannada_reverse_char_index
}


# Hebrew

hebrew_alphabets = sorted(['\t','\n',' '] + [chr(alpha) for alpha in range(1424, 1535)])
hebrew_char_index = dict([(char, i) for i, char in enumerate(hebrew_alphabets)])
hebrew_reverse_char_index = dict((i, char) for char, i in hebrew_char_index.items())

hebrew = {
    'vocab_size' : 114,
    'alphabets' : hebrew_alphabets,
    'char_index' : hebrew_char_index,
    'reverse_char_index' : hebrew_reverse_char_index
}