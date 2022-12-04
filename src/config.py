def prepare_input_output_dic(input_characters, target_characters):
    
    char_index = dict([(char, i) for i, char in enumerate(sorted([' '] + input_characters))])
    reverse_char_index = dict((i, char) for char, i in char_index.items())
    
    input_dic = {
        'vocab_size' : len(input_characters) + 1,
        'alphabets' : input_characters,
        'char_index' : char_index,
        'reverse_char_index' : reverse_char_index,
        'max_length' : 20
    }
    
    char_index = dict([(char, i) for i, char in enumerate(sorted(['\t','\n',' '] + target_characters))])
    reverse_char_index = dict((i, char) for char, i in char_index.items())
    
    output_dic = {
        'vocab_size' : len(target_characters) + 3,
        'alphabets' : target_characters,
        'char_index' : char_index,
        'reverse_char_index' : reverse_char_index,
        'max_length' : 20
    }
    
    return input_dic, output_dic
    

# English

english_alphabets = sorted(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))

# hindi

hindi_alphabets = sorted([chr(alpha) for alpha in range(2304, 2432)])

# Bangla

bangla_alphabets = sorted([chr(alpha) for alpha in range(2432, 2559)])

# Tamil

tamil_alphabets = sorted([chr(alpha) for alpha in range(2944, 3071)])

# Kannada

kannada_alphabets = sorted([chr(alpha) for alpha in range(3200, 3327)])

# Hebrew

hebrew_alphabets = sorted([chr(alpha) for alpha in range(1424, 1535)])