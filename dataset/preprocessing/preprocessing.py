import pandas as pd
import re
import string

from nltk.stem import PorterStemmer
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from num2words import num2words
from langdetect import detect

from tqdm import tqdm

research_paper_stopwords = ['introduction', 'abstract', 'section', 'edition', 'chapter',
                            'copyright', 'preprint', 'figure']
english_stopwords = stopwords.words("english")
cachedStopWords = set(english_stopwords + research_paper_stopwords)

#covid_key_words=["2019", "novel coronavirus", "covid", "ncov", "wuhan"
 #                "sars-cov-2", "coronavirus", "severe acute respiratory syndrome",
 #                "corona"]  --- makla sam jer ovaj ncov ce vratit paper koji ima npr 'uncover', a 2019. ako je citiran neki rad npr. (Markotic, 2019)
covid_key_words=["novel coronavirus", "covid", "wuhan"
                 "sars-cov-2", "coronavirus", "severe acute respiratory syndrome",
                 "corona"]

#covid_key_words = [
#    'covid',
#    'corona',
#    'coronavirus',
#    'sars-cov-2',
#    'severe acute respiratory syndrome'
#]

##if q=True, don't remove paragraphs with less than 2 sentences

def contains_key_words(text, key_words):
    return any([x in text for x in covid_key_words])


def preprocess_data(texts, q):  # requires json format
    sentences = []
    temp = texts
    for paragraph in tqdm(texts):
        if 'text' in paragraph and not q and contains_key_words(paragraph['text'].lower(), covid_key_words):
            d = {}
            stem_sentence = []
            for k in paragraph.keys():
                if k != 'text':
                    d.update({k: paragraph[k]})
                else:
                    d.update({k: paragraph[k]})
                    stem_sentence = word_stem(paragraph[k], q)
                    d['preprocessed_text'] = stem_sentence
            sentences.append(d)
            temp = pd.DataFrame(sentences)
        if 'text' in paragraph and q:
            d = {}
            stem_sentence = []
            for k in paragraph.keys():
                if k != 'text':
                    d.update({k: paragraph[k]})
                else:
                    d.update({k: paragraph[k]})
                    stem_sentence = word_stem(paragraph[k], q)
                    d['preprocessed_text'] = stem_sentence
            sentences.append(d)
            temp = pd.DataFrame(sentences)
    return filter_by_language(temp[temp['preprocessed_text'] != ''])


def word_stem(sentences, q):
    porter = PorterStemmer()
    stem_sentences = []
    new_sentences = sent_tokenize(sentences)
    sentences = []
    for sent in new_sentences:
        word_tokens = word_tokenize(sent)
        if word_tokens[-1] == '?':
            continue
        else:
            sentences.append(sent)
    if not q:
        if len(sentences) < 2:
            return ''
    for sent in sentences:
        stem_sentence = []
        word_tokens = word_tokenize(sent)
        word_tokens = remove_punctuation(word_tokens)
        word_tokens = num_to_word(word_tokens)
        word_tokens = remove_too_short(word_tokens, q)
        word_tokens = remove_stop_words(word_tokens)
        if (len(word_tokens) > 0):
            for word in word_tokens:
                stem_sentence.append(porter.stem(word))
                stem_sentence.append(' ')
        if (len(stem_sentence) > 0):
            stem_sentences.append(''.join(stem_sentence))
    return '. '.join(stem_sentences) + '.'


def remove_too_short(tokens, q):
    new_tokens = []
    if not q:
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
        if w.isnumeric() and flag:
            new_tokens.append(num2words(w))
        elif bool(patt.match(w)):
            #            new_tokens.append(num2words(w.replace('.',''),'ordinal_num'))  #if we need 15. = fifteenth
            new_tokens.append(num2words(w.replace('.', '')))
        else:
            new_tokens.append(w)
    return new_tokens


def remove_stop_words(tokens):
    tokens = [w for w in tokens if w not in cachedStopWords]
    return tokens


def preprocess_query(query, q):
    return list(preprocess_data([{'text': query}], q)['preprocessed_text'].replace('.', ''))


def filter_by_key_words(data, key_words, column_name='text'):
    """
    Keeps rows of the pandas DataFrame (data) that contain any of
    given key words as a substring in a the column defined by column_name

    e.g. row1[text] = 'Why would you do that, why??', row2[text] = 'Covid 19', key_words = ['cov'],
    it would keep row2, and not row1, because 'cov' is a substring of 'Covid'

    Args:
        data: pandas DataFrame
        key_words:
        column_name:

    Returns:
    data with only rows that contain any of the given key words as a substring
    """
    pattern = r'|'.join(key_words)
    return data[data[column_name].str.lower().str.contains(pattern)]


def filter_by_language(data, column='text', lan='en'):
    """
    Keeps rows of the given DataFrame(data) that have the given column value written in the given language.

    Args:
        data: pandas DataFrame
        column:
        lan:

    Returns:
    pd DataFrame with rows containing texts in the given language
    """
    wanted = []
    for index, row in tqdm(data.iterrows()):
        try:
            if detect(row[column]) == lan:
                wanted.append(index)
        except:
            pass
    return data.loc[wanted]
