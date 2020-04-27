import pandas as pd
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")

#TODO: Add removing punctuation, convert numbers to words, removing stop words ...
def preprocess_data(texts):
    sentences = []
    for sentence in texts:
        d = {}
        stem_sentence = []
        for k in sentence.keys():
            if k!='text':
                d.update({k: sentence[k]})
            else:
                filtered_sent=remove_stop_words(sentence[k])
                stem_sentence = word_stem(filtered_sent)
                d.update({k: stem_sentence})
        sentences.append(d)
    return pd.DataFrame(sentences) 

def word_stem(sentence):
    porter = PorterStemmer()
    stem_sentence = []
    word_tokens = word_tokenize(sentence)
    for word in word_tokens:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(' ')
    stem_sentence = ''.join(stem_sentence)
    return stem_sentence

def remove_punctuation(sentence):
    pass

def num_to_word(sentece):
    pass

def remove_stop_words(sentence):
    sentence=' '.join([word for word in sentence.split() if word.lower() not in cachedStopWords])
    return sentence
