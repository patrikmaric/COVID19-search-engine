from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec, Doc2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing.preprocessing import preprocess_data
from preprocessing.preprocessing import word_stem

import pandas as pd

import numpy as np

# Enters paragraph; make sentences and words to feed W2V
#from data import CovidDataLoader
from dataset.data import CovidDataLoader
from query_model.queries import QueryEngine

from settings import data_root_path




class W2VQueryEngine(QueryEngine):

    def __init__(self):
        super().__init__()

    def fit(self, corpus):
        self.corpus = corpus  # expect pd.DataFrame
        self.__build_w2v()
        self.__build_paragraph_embeddings()

    def run_query(self, query, n=5):
        # TODO; check this
        query_vector = self.get_paragraph_embedding(query)

        n1 = np.linalg.norm(query_vector)
        qvn = np.divide(query_vector, n1)
        n2 = np.linalg.norm(self.paragraph_vectors, axis=1)
        pvn = np.divide(self.paragraph_vectors, n2.reshape(-1, 1))

        similarities = qvn.dot(pvn.T)
        return self.__create_query_result(query, similarities, n)

    def __build_w2v(self):
        tok_corpus = []
        for paragraph in self.corpus['text']:
            senten = sent_tokenize(paragraph)
            for sent in senten:
                sent = word_stem(sent)
                tok_corpus.append(word_tokenize(sent))
        # building vocab
        self.w2v = Word2Vec(tok_corpus, min_count=1, size=50, workers=3, window=3, sg=1)

    def __build_paragraph_embeddings(self):
        vectors = []
        for element in self.corpus['text']:
            element_vec = self.get_paragraph_embedding(element).reshape(1, -1)
            vectors.append(element_vec[0])

        self.paragraph_vectors = np.array(vectors)

    ##Add up word2vec
    def get_paragraph_embedding(self, paragraph):
        sentences = sent_tokenize(paragraph)
        word_list = []
        for sent in sentences:
            sent = word_stem(sent)
            word_list.append(word_tokenize(sent))
        words = []
        for sublist in word_list:
            for item in sublist:
                words.append(item)
        result_vec = np.zeros(np.shape(self.w2v[list(self.w2v.wv.vocab.keys())[0]]))
        for word in words:
            if word in self.w2v.wv.vocab.keys():
                result_vec += np.array(self.w2v[word])
        return result_vec

    def __create_query_result(self, query, similarities, n):
        result = {
            'id': self.corpus['paper_id'],
            'query': query,
            'text': self.corpus['text'],
            'sim': similarities,
        }

        result = pd.DataFrame(result).sort_values(by='sim', ascending=False)[:n]

        return result[result['sim'] > 0]

if __name__ == '__main__':
    article_paths = CovidDataLoader.load_articles_paths(data_root_path)

    abstracts = CovidDataLoader.load_data(article_paths, key='body_text',offset=0, limit=10, load_sentences=False, preprocess=False)
    query1 = word_stem("Main risk factors for covid19")
    query2 = word_stem("Does smoking increase risks when having covid19?")
    query3 = word_stem("What is the mortality rate of covid19?")

    query_engine = W2VQueryEngine()
    query_engine.fit(abstracts)
    results = query_engine.run_query(query1)

