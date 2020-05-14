from nltk.corpus import stopwords

from dataset.data import CovidDataLoader, abstract_keys
#from data import CovidDataLoader, abstract_keys

from query_model.transformers.bm25 import BM25Transformer
from settings import data_root_path

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle

import pandas as pd


# TODO: short(wrongly separated) sentences are the current problem...


class QueryEngine():

    def __init__(self):
        self.corpus = None

    def run_query(self, query, n=5):
        """
        Runs the given query, returns max n most similar documents from the corpus on which the model was built.

        Args:
            query: query to run
            n: max number of results returned

        Returns:
            n(or less) most similar documents from the corpus on which the model was built
        """
        pass

    def fit(self, corpus):
        """
        Builds the query engine on the given corpus.

        Args:
            corpus: list of documents to build the model on
            document_ids: optional, if given it will associate the given id's to each document given in the corpus
        """
        pass

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
            'id': self.corpus['paper_id'],
            'query': query * len(self.corpus),
            'text': self.corpus['text'],
            'sim': sims,
        }

        result = pd.DataFrame(result).sort_values(by='sim', ascending=False)[:n]

        return result[result['sim'] > 0]

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


class BOWQueryEngine(QueryEngine):

    def __init__(self, count_vectorizer, transformer):
        super().__init__()
        self.cv = count_vectorizer
        self.transformer = transformer

    def run_query(self, query, n=5):
        if self.corpus is None:
            raise AttributeError('Model not built jet, please call the fit method before running queries!')

        if type(query) == str:
            query = [query]
        query_word_count_vector = self.cv.transform(query)
        query_vector = self.transformer.transform(query_word_count_vector)
        similarities = query_vector.dot(self.corpus_vector.T)  # TODO: check if this already sorts values
        return self._QueryEngine__create_query_result(query, similarities, n)

    def fit(self, corpus):
        self.corpus = corpus
        word_count_vector = self.cv.fit_transform(corpus['text'])
        self.transformer.fit(word_count_vector)
        self.corpus_vector = self.transformer.transform(word_count_vector)


if __name__ == '__main__':
    stop_words = set(stopwords.words('english'))

    article_paths = CovidDataLoader.load_articles_paths(data_root_path)
    abstracts = CovidDataLoader.load_data(article_paths, key='abstract', offset=60000, limit=1000, keys=abstract_keys, load_sentences=True)


    cv = CountVectorizer(stop_words=stop_words)

    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    transformer = BM25Transformer()

    query_engine = BOWQueryEngine(cv, transformer)
    query_engine.fit(abstracts)

    # query_engine.save('./', 'abstracts_query_engine2')
    # query_engine2 = QueryEngine.load('abstracts_query_engine.dat')

    # query = ['similar health treatment']
    query = 'LDH'
    query_result = query_engine.run_query(query)
