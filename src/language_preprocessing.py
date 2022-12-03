import re
from pathlib import Path
import xml.etree.ElementTree as ET

def create_Alphabet_Dict(alphabets) -> dict:
    return dict([(char, i) for i, char in enumerate(alphabets)])

def clean_English_Vocab(line: str) -> list:
    non_eng_letters_regex = re.compile('[^a-zA-Z ]')
    line = line.replace('-', ' ').replace(',',' ').upper()
    line = non_eng_letters_regex.sub('', line)
    return line.split()

def clean_Vocab(line:str, alphabets) -> list:
    hindi_dic = create_Alphabet_Dict(alphabets)
    line = line.replace('-', ' ').replace(',', ' ')
    cleaned_line = ''
    for char in line:
        if char in hindi_dic or char == ' ':
            cleaned_line += char
    return cleaned_line.split()

# U+0080 ... U+00FF: Latin-1 Supplement
# U+0100 ... U+017F: Latin Extended-A
# U+0180 ... U+024F: Latin Extended-B
# U+0250 ... U+02AF: IPA Extensions
# U+02B0 ... U+02FF: Spacing Modifier Letters
# U+0300 ... U+036F: Combining Diacritical Marks
# U+0370 ... U+03FF: Greek and Coptic
# U+0400 ... U+04FF: Cyrillic
# U+0500 ... U+052F: Cyrillic Supplement
# U+0530 ... U+058F: Armenian
# U+0590 ... U+05FF: Hebrew
# U+0600 ... U+06FF: Arabic
# U+0700 ... U+074F: Syriac
# U+0750 ... U+077F: Arabic Supplement
# U+0780 ... U+07BF: Thaana
# U+07C0 ... U+07FF: NKo
# U+0800 ... U+083F: Samaritan
# U+0900 ... U+097F: Devanagari
# U+0980 ... U+09FF: Bengali
# U+0A00 ... U+0A7F: Gurmukhi
# U+0A80 ... U+0AFF: Gujarati
# U+0B00 ... U+0B7F: Oriya
# U+0B80 ... U+0BFF: Tamil
# U+0C00 ... U+0C7F: Telugu
# U+0C80 ... U+0CFF: Kannada
# U+0D00 ... U+0D7F: Malayalam

def clean_English_Vocab(line: str) -> list:
    non_eng_letters_regex = re.compile('[^a-zA-Z ]')
    line = line.replace('-', ' ').replace(',',' ').upper()
    line = non_eng_letters_regex.sub('', line)
    return line.split()