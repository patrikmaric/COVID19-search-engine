from nltk.corpus import stopwords

from dataset.data import CovidDataLoader, abstract_keys
#from data import CovidDataLoader, abstract_keys

from query_model.transformers.bm25 import BM25Transformer
from settings import data_root_path

from sklearn.feature_extraction.text import CountVectorizer

import pickle

import pandas as pd


# TODO: short(wrongly separated) sentences are the current problem...


class QueryEngine():

    def __init__(self, count_vectorizer, transformer):
        self.cv = count_vectorizer
        self.transformer = transformer
        self.corpus = None
        self.ids = None

    def fit(self, corpus, document_ids=None):
        """
        Builds the query engine on the given corpus.

        Args:
            corpus: list of documents to build the model on
            document_ids: optional, if given it will associate the given id's to each document given in the corpus
        """
        self.ids = document_ids
        self.corpus = corpus
        word_count_vector = self.cv.fit_transform(corpus)
        self.transformer.fit(word_count_vector)
        self.corpus_tf_idf_vector = self.transformer.transform(word_count_vector)

    def __create_query_result(self, query, similarities, n):
        """

        Args:
            similarities: sparse matrix containing cosine similarities between the query vector and documents from corpus
            n: number of most similar documents to include in the result

        Returns:
            pandas DataFrame containing query, document, similarity
        """
        sims = similarities.toarray()[0]

        result = {
            'query': query * len(sims),
            'text': self.corpus,
            'sim': sims,
        }
        if self.ids:
            result.update({'id': self.ids})

        result = pd.DataFrame(result).sort_values(by='sim', ascending=False)[:n]

        return result[result['sim'] > 0]

    def run_query(self, query, n=5):
        """
        Runs the given query, returns max n most similar documents from the corpus on which the model was built.

        Args:
            query: query to run
            n: max number of results returned

        Returns:
            n(or less) most similar documents from the corpus on which the model was built
        """
        if self.corpus is None:
            raise AttributeError('Model not built jet, please call the fit method before running queries!')

        if type(query) == str:
            query = [query]
        query_word_count_vector = self.cv.transform(query)
        query_tf_idf_vector = self.transformer.transform(query_word_count_vector)
        similarities = query_tf_idf_vector.dot(self.corpus_tf_idf_vector.T)  # TODO: check if this already sorts values
        return self.__create_query_result(query, similarities, n)

    def save(self, dir_path, name):
        """
        Serializes the object to file(name.dat) to the directory defined by the path.

        Args:
            dir_path: path of the directory to save the object to
            name: name of the file (without extension)
        """
        pickle_path = dir_path + name + '.dat'
        print('Writing object to %s' % pickle_path)
        with open(pickle_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(pickle_path):
        """
        Loads(de-serializes) QueryEngine object from the file at the given path.

        Args:
            pickle_path: path to QueryEngine pickle

        Returns:
            QueryEngine object
        """
        with open(pickle_path, 'rb') as f:
            query_engine = pickle.load(f)
            if type(query_engine) != QueryEngine:
                raise ValueError('Path to non QueryEngine object!')
            return query_engine


if __name__ == '__main__':
    stop_words = set(stopwords.words('english'))

    article_paths = CovidDataLoader.load_articles_paths(data_root_path)
    abstracts = CovidDataLoader.load_data(article_paths, key='abstract', limit=1, keys=abstract_keys, load_sentences=True)
    
    corpus = list(abstracts['text'])
    paper_ids = list(abstracts['paper_id'])

    cv = CountVectorizer(stop_words=stop_words)

    # transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    transformer = BM25Transformer()

    query_engine = QueryEngine(cv, transformer)
    query_engine.fit(corpus, paper_ids)

    query_engine.save('./', 'abstracts_query_engine2')
    #query_engine2 = QueryEngine.load('abstracts_query_engine.dat')

    # query = ['similar health treatment']
    query = 'LDH'
    #query_result = query_engine2.run_query(query)
