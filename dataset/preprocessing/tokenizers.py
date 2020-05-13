from nltk import WordNetLemmatizer, word_tokenize, PorterStemmer


# https://stackoverflow.com/questions/47423854/sklearn-adding-lemmatizer-to-countvectorizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


class StemmTokenizer(object):
    def __init__(self):
        self.stemmer = PorterStemmer()

    def __call__(self, articles):
        return [self.stemmer.stem(t) for t in word_tokenize(articles)]
