import pandas as pd
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords


#TODO: Add removing punctuation, convert numbers to words, removing stop words ...
#TODO: if we have word-number, remove -, it happens in covid-19 and it will be removed in remove_punctuation
def preprocess_data(texts):
    sentences = []
    for sentence in texts:
        d = {}
        stem_sentence = []
        for k in sentence.keys():
            if k!='text':
                d.update({k: sentence[k]})
            else:
                stem_sentence = word_stem(sentence[k])
                d.update({k: stem_sentence})
        sentences.append(d)
    return pd.DataFrame(sentences) 

def word_stem(sentence):
    porter = PorterStemmer()
    stem_sentence = []
    word_tokens = word_tokenize(sentence)
    word_tokens = remove_stop_words(word_tokens)
    word_tokens = remove_punctuation(word_tokens)
    for word in word_tokens:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(' ')
    stem_sentence = ''.join(stem_sentence)
    return stem_sentence

def remove_punctuation(tokens):
    tokens_new = []
    for w in tokens:
        if w.isalnum():
            tokens_new.append(w)
    return tokens_new

def num_to_word(sentece):
    pass

def remove_stop_words(tokens):
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return tokens