import pandas as pd
from nltk.stem import PorterStemmer
from nltk import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from num2words import num2words 
import re
import string  

cachedStopWords = stopwords.words("english")


##if q=True, don't remove paragraphs with less than 2 sentences

def preprocess_data(texts, q):  #requires json format
    sentences = []
    for paragraph in texts:
        d = {}
        stem_sentence = []
        for k in paragraph.keys():
            if k != 'text':
                d.update({k: paragraph[k]})
            else:
                #                filtered_sent=remove_stop_words(sentence[k])
                #                stem_sentence = word_stem(filtered_sent)
                #                d.update({k: stem_sentence})
                d.update({k: paragraph[k]})
                stem_sentence = word_stem(paragraph[k],q)
                d['preprocessed_text'] = stem_sentence
        sentences.append(d)
        temp = pd.DataFrame(sentences)
    return temp[temp['preprocessed_text'] != '']


def word_stem(sentences,q):
    porter = PorterStemmer()
    stem_sentences = []
    sentences = sent_tokenize(sentences)
    if not q:
        if len(sentences) < 2:
            return ''
    for sent in sentences:
        stem_sentence = []
        word_tokens = word_tokenize(sent)
        word_tokens = remove_punctuation(word_tokens)
        word_tokens = num_to_word(word_tokens)
        word_tokens = remove_too_short(word_tokens)
        word_tokens = remove_stop_words(word_tokens)
        for word in word_tokens:
            stem_sentence.append(porter.stem(word))
            stem_sentence.append(' ')
        stem_sentences.append(''.join(stem_sentence))
    return '. '.join(stem_sentences) + '.'


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
        flag = True
        for ch in w:
            if ch not in string.printable:
                flag = False
        if w.isnumeric() and  flag:
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

def preprocess_query(query, q):
    return list(preprocess_data([{'text': query}],q)['preprocessed_text'].replace('.',''))

"""def remove_stop_words(sentence):
    sentence = ' '.join([word for word in sentence.split() if word.lower() not in cachedStopWords])
    return sentence
"""
