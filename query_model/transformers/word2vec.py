from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np
from dataset.preprocessing.preprocessing import preprocess_query

import pandas as pd



#from data import CovidDataLoader
from dataset.data import CovidDataLoader
from query_model.queries import QueryEngine

from settings import data_root_path


#Expects preprocessed text -no preprocessing done here;
#INPUT: abstracts --- pd.Dataframe type, query --- preprocessed string
class W2VQueryEngine(QueryEngine):

    def __init__(self):
        super().__init__()

    def fit(self, corpus, text_column='preprocessed_text'):
        self.corpus = corpus
        self.__build_w2v(text_column)
        self.__build_paragraph_embeddings(text_column)

    def run_query(self, query, n=5, q=True):
        query = preprocess_query(query, q)[0]
        query_vector = self.get_paragraph_embedding(query)

        n1 = np.linalg.norm(query_vector)
        qvn = np.divide(query_vector, n1)
        n2 = np.linalg.norm(self.paragraph_vectors, axis=1)
        pvn = np.divide(self.paragraph_vectors, n2.reshape(-1, 1))

        similarities = qvn.dot(pvn.T)
        return self.__create_query_result(query, similarities, n)

    def __build_w2v(self, text_column):
        tok_corpus = []
        for paragraph in self.corpus[text_column]:
            if paragraph != '':
                for sent in paragraph:
                    tok_corpus.append(word_tokenize(sent))
        # building vocab
        self.w2v = Word2Vec(tok_corpus, min_count=1, size=50, workers=3, window=3, sg=1)
        

    def __build_paragraph_embeddings(self, text_column):
        vectors = []
        for element in self.corpus[text_column]:
            element_vec = self.get_paragraph_embedding(element).reshape(1, -1)
            vectors.append(element_vec[0])

        self.paragraph_vectors = np.array(vectors)

    ##Add up word2vec
    def get_paragraph_embedding(self, paragraph):
        word_list = []
        for sent in paragraph:
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

    abstracts = CovidDataLoader.load_data(article_paths, key='body_text',offset=0, limit=100, load_sentences=False, preprocess=True, q=False)
    abstracts['preprocessed_text'] = abstracts['preprocessed_text'].str.replace('.','')
    
    inserted_query = "Main risk factors for covid-19"
    inserted_query2 = "Does smoking increase risks when having covid19?"
    inserted_query3 = "What is the mortality rate of covid19?"


#    query_engine = W2VQueryEngine()
#    
# 
#    query_engine.fit(abstracts)
#    results = query_engine.run_query(inserted_query)

