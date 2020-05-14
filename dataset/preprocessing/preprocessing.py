import pandas as pd
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from num2words import num2words 
import re

cachedStopWords = stopwords.words("english")

def preprocess_data(texts):
    sentences = []
    for sentence in texts:
        d = {}
        stem_sentence = []
        for k in sentence.keys():
            if k != 'text':
                d.update({k: sentence[k]})
            else:
                #                filtered_sent=remove_stop_words(sentence[k])
                #                stem_sentence = word_stem(filtered_sent)
                #                d.update({k: stem_sentence})
                stem_sentence = word_stem(sentence[k])
                d.update({k: stem_sentence})
        sentences.append(d)
    return pd.DataFrame(sentences)


def word_stem(sentence):
    porter = PorterStemmer()
    stem_sentence = []
    word_tokens = word_tokenize(sentence)
    word_tokens = remove_punctuation(word_tokens)
    word_tokens = num_to_word(word_tokens)
    word_tokens = remove_too_short(word_tokens)
    word_tokens = remove_stop_words(word_tokens)
    for word in word_tokens:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(' ')
    stem_sentence = ''.join(stem_sentence)
    return stem_sentence


def remove_too_short(tokens):
    new_tokens = []
    if len(tokens) < 5:
        return new_tokens
    for w in tokens:
        if len(w) > 2:
            new_tokens.append(w)
    return new_tokens

            
def remove_punctuation(tokens):
    patt1 = re.compile("^\w+-\d+$")
    patt2 = re.compile("^\d+\.$")
    tokens_new = []
    for w in tokens:
        if w.isalnum() or not bool(patt1.match(w)) or not bool(patt2.match(w)):
            tokens_new.append(w)    
    return tokens_new


def num_to_word(tokens):
    new_tokens = []
    patt = re.compile("^\d+\.$")
    for w in tokens:
        if w.isnumeric():
            new_tokens.append(num2words(w))
        elif bool(patt.match(w)):
#            new_tokens.append(num2words(w.replace('.',''),'ordinal_num'))  #if we need 15. = fifteenth
            new_tokens.append(num2words(w.replace('.','')))
        else:
            new_tokens.append(w)
    return new_tokens


def remove_stop_words(tokens):
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return tokens


"""def remove_stop_words(sentence):
    sentence = ' '.join([word for word in sentence.split() if word.lower() not in cachedStopWords])
    return sentence
"""
