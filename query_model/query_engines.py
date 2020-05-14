import pickle

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import sent_tokenize, word_tokenize

from dataset.preprocessing.preprocessing import preprocess_query
from query_model.utils import BERT_sentence_embeddings


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

    def fit(self, corpus, text_column='preprocessed_text'):
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

    def fit(self, corpus, text_column='preprocessed_text'):
        self.corpus = corpus
        word_count_vector = self.cv.fit_transform(corpus['text'])
        self.transformer.fit(word_count_vector)
        self.corpus_vector = self.transformer.transform(word_count_vector)


# Expects preprocessed text -no preprocessing done here;
# INPUT: abstracts --- pd.Dataframe type, query --- preprocessed string
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


# requires preprocessed text; shouldn't load sentences
# INPUT: pd.Dataframe - abstracts; query - string
class D2VQueryEngine(QueryEngine):

    def __init__(self):
        super().__init__()

    def fit(self, corpus, text_column='preprocessed_text'):
        self.corpus = corpus
        self.__build_d2v(text_column)
        self.__build_paragraph_embeddings(text_column)

    def run_query(self, query, n=5, q=True):
        query = preprocess_query(query, q)[0]
        query_tokens = []
        senten = sent_tokenize(query)
        for sent in senten:
            tok = word_tokenize(sent)
            tok.remove('.')
            query_tokens += tok
        query_vector = self.d2v.infer_vector(query_tokens).reshape(1, -1)
        n1 = np.linalg.norm(query_vector)
        qvn = np.divide(query_vector, n1)
        n2 = np.linalg.norm(self.paragraph_vectors, axis=1)
        pvn = np.divide(self.paragraph_vectors, n2.reshape(-1, 1))

        similarities = qvn.dot(pvn.T)[0]
        return self.__create_query_result(query, similarities, n)

    def __build_d2v(self, text_column):
        tok_corpus = []
        cnt = 0
        for paragraph in self.corpus[text_column]:
            cnt += 1
            j = str(cnt)
            sentn = sent_tokenize(paragraph)
            par_words = []
            tags = []
            for i, sent in enumerate(sentn):
                tags.append(j + str(i))
                par_words.append(word_tokenize(sent))
                for element in par_words:
                    if '.' in element:
                        element.remove('.')
            tok_corpus += [TaggedDocument(words=par_words[j], tags=tags[j]) for j in range(len(par_words))]
        # building vocab
        self.d2v = Doc2Vec(dm=0, vector_size=300, min_count=5, negative=5, hs=0, sample=0, epochs=400,
                           window=15)  # it was 2, but it says that it works better with min_count=5
        self.d2v.build_vocab(tok_corpus)
        self.d2v.train(tok_corpus, total_examples=self.d2v.corpus_count, epochs=self.d2v.epochs)

    def __build_paragraph_embeddings(self, text_column):
        vectors = []
        for element in self.corpus[text_column]:
            element_tokens = []
            senten = sent_tokenize(element)
            for sent in senten:
                words = word_tokenize(sent)
                if '.' in words:
                    words.remove('.')
                element_tokens += words
            element_vec = self.d2v.infer_vector(element_tokens).reshape(1, -1)
            vectors.append(element_vec[0])
        self.paragraph_vectors = np.array(vectors)

    def __create_query_result(self, query, similarities, n):
        result = {
            'id': self.corpus['paper_id'],
            'query': query,
            'text': self.corpus['text'],
            'sim': similarities,
        }
        result = pd.DataFrame(result).sort_values(by='sim', ascending=False)[:n]

        return result[result['sim'] > 0]


class BERTQueryEngine(QueryEngine):

    def __init__(self):
        super().__init__()

    def fit(self, corpus, text_column='preprocessed_text'):
        self.corpus = corpus
        self.corpus_embeddings = BERT_sentence_embeddings(corpus, text_column, query=False)

    def run_query(self, query, n=5):
        if self.corpus is None:
            raise AttributeError('Model not built yet, please call the fit method before running queries!')

        assert type(query) == str

        query_embedding = BERT_sentence_embeddings(query, query=True)
        similarities = np.dot(self.corpus_embeddings, query_embedding.T)  # TODO: check if this already sorts values

        return self.__create_query_result(query, similarities, n)

    def __create_query_result(self, query, similarities, n):
        """

        Args:
            similarities: sparse matrix containing cosine similarities between the query vector and documents from corpus
            n: number of most similar documents to include in the result

        Returns:
            pandas DataFrame containing query, document, similarity
        """

        result = {
            'id': self.corpus['paper_id'],
            'query': [query] * len(self.corpus),
            'text': self.corpus['text'],
            'sim': np.squeeze(similarities)
        }

        result = pd.DataFrame(result).sort_values(by='sim', ascending=False)[:n]

        # return result
        return result[result['sim'] > 0]
