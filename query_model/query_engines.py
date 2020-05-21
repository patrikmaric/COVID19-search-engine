import pickle

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import sent_tokenize, word_tokenize

from dataset.preprocessing.preprocessing import preprocess_query
#from preprocessing.preprocessing import preprocess_query
from data import CovidDataLoader, body_text_keys
from settings import data_root_path
from tqdm import tqdm
from query_model.utils import BERT_sentence_embeddings
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

epsilon = 0.0000000001

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

    def __init__(self, w2v_params):
        super().__init__()
        self.w2v_params = w2v_params

    def fit(self, corpus, text_column='preprocessed_text', pooling='average'):
        self.corpus = corpus
        self.tf_idf = TfidfVectorizer(smooth_idf=True, use_idf=True).fit(corpus[text_column])
        self.__build_w2v(text_column)
        self.__build_paragraph_embeddings(text_column, pooling)

    def run_query(self, query, n=5, q=True):
        preprocessed_query = preprocess_query(query, q)[0]
        query_vector = self.get_paragraph_embedding(preprocessed_query)
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
                tok = word_tokenize(paragraph)
                while '.' in tok:
                    tok.remove('.')
                tok_corpus.append(tok)
        # building vocab
        self.w2v = Word2Vec(tok_corpus, **self.w2v_params)

    def __build_paragraph_embeddings(self, text_column, pooling):
        vectors = []
        for element in self.corpus[text_column]:
            element_vec = self.get_paragraph_embedding(element, pooling).reshape(1, -1)
            vectors.append(element_vec[0])

        self.paragraph_vectors = np.array(vectors)

    def get_paragraph_embedding(self, paragraph, pooling):
        word_list = word_tokenize(paragraph)
        
        while '.' in word_list:
            word_list.remove('.')
        
        if pooling=='average':
            result_vec = np.zeros(np.shape(self.w2v[list(self.w2v.wv.vocab.keys())[0]])) + epsilon
            cnt = 0
            for word in word_list:
                if word in self.w2v.wv.vocab.keys():
                    cnt += 1
                    result_vec += np.array(self.w2v[word])
            if cnt > 0:
                return result_vec/cnt
            else:
                return result_vec
            
        if pooling=='weighted':
            result_vec = np.zeros(np.shape(self.w2v[list(self.w2v.wv.vocab.keys())[0]])) + epsilon
            cnt = 0
            
            df=dict(Counter(paragraph.split()).most_common())
            norm=sum([a**2 for a in list(df.values()) ])
            normalised_df = {k: v/norm for k, v in df.items()}
            idf=dict(zip(self.tf_idf.get_feature_names(), self.tf_idf.idf_))
            
            for word in word_list:
                if word in self.w2v.wv.vocab.keys():
                    cnt += 1
                    result_vec += np.array(self.w2v[word])*normalised_df[word]*idf['word']
            if cnt > 0:
                return result_vec/cnt
            else:
                return result_vec


# requires preprocessed text; shouldn't load sentences
# INPUT: pd.Dataframe - abstracts; query - string
class D2VQueryEngine(QueryEngine):

    def __init__(self, d2v_params):
        super().__init__()
        self.d2v_params = d2v_params

    def fit(self, corpus, text_column='preprocessed_text'):
        self.corpus = corpus
        self.__build_d2v(text_column)
        self.__build_paragraph_embeddings(text_column)

    def run_query(self, query, n=10, q=True):
        preprocessed_query = preprocess_query(query, q)[0]
        query_tokens = []
        senten = sent_tokenize(preprocessed_query)
        for sent in senten:
            tok = word_tokenize(sent)
            tok.remove('.')
            query_tokens += tok
        
        query_vector = self.d2v.infer_vector(query_tokens, steps=400, alpha=0.025).reshape(1, -1)
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
        self.d2v = Doc2Vec(dm=0, **self.d2v_params)
        self.d2v.build_vocab(tok_corpus)
        self.d2v.train(tok_corpus, total_examples=self.d2v.corpus_count, epochs=self.d2v.epochs)
        
    def __build_paragraph_embeddings(self, text_column):
        vectors = []
        for element in tqdm(self.corpus[text_column]):
            element_tokens = []
            senten = sent_tokenize(element)
            for sent in senten:
                words = word_tokenize(sent)
                if '.' in words:
                    words.remove('.')
                element_tokens += words
            element_vec = self.d2v.infer_vector(element_tokens, steps=30, alpha=0.025).reshape(1, -1)
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

if __name__ == '__main__':
    article_paths = CovidDataLoader.load_articles_paths(data_root_path)
    texts = CovidDataLoader.load_data(article_paths, key='body_text', offset=0, limit=None, keys=body_text_keys,
                                      load_sentences=False, preprocess=True)
    abstracts = CovidDataLoader.load_data(article_paths, key='abstract', offset=0, limit=None, keys=body_text_keys,
                                      load_sentences=False, preprocess=True)
    corpus = pd.concat([texts,abstracts])
    nmb_par = len(corpus)
    print('Number of paragraphs is:', nmb_par)
    params = {
        'min_count': 5, #5, #w2v
        'vector_size': 300,
        #'workers': 3,  #v2w
        'window': 15, #3, w2v
        'hs': 0,
        'negative': 20,
        'sample': 0, #ne w2v
        'epochs': 30, #ne w2v
        'alpha': 0.025,
        'min_alpha': 0.00025
    }
    print('D2V...')
    query_engine = D2VQueryEngine(params)
    query_engine.fit(corpus)
    query_engine.save('/home/nikolina/','d2v')
    print("Incubation period of Covid19?")
    result = query_engine.run_query("Incubation period of Covid19?")
