from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing.preprocessing import preprocess_data
from preprocessing.preprocessing import word_stem

import pandas as pd

import numpy as np

# Enters paragraph; make sentences and words to feed W2V
from data import CovidDataLoader
#from dataset.data import CovidDataLoader
from query_model.queries import QueryEngine

from settings import data_root_path


#TODO: CHECK EVERYTHING
class D2VQueryEngine(QueryEngine):

    def __init__(self):
        super().__init__()

    def fit(self, corpus):
        self.corpus = corpus  # expect pd.DataFrame
        self.__build_d2v()
        self.__build_paragraph_embeddings()

    def run_query(self, query, n=5):
        query_tokens = []
        senten = sent_tokenize(query)
        for sent in senten:
            sent = word_stem(sent)
            query_tokens += word_tokenize(sent)
        query_vector = self.d2v.infer_vector(query_tokens).reshape(1,-1)
        n1 = np.linalg.norm(query_vector)
        qvn = np.divide(query_vector, n1)
        n2 = np.linalg.norm(self.paragraph_vectors, axis=1)
        pvn = np.divide(self.paragraph_vectors, n2.reshape(-1, 1))

        similarities = qvn.dot(pvn.T)[0]
        return self.__create_query_result(query, similarities, n)

    def __build_d2v(self):
        tok_corpus = []
        tags = []
        words = []
        for i,paragraph in enumerate(self.corpus['text']):
            tags += str(i)
            sentn = sent_tokenize(paragraph)
            par_words = []
            for sent in sentn:
                par_words += word_tokenize(word_stem(sent))
            words.append(par_words)
        tok_corpus+=[TaggedDocument(words=words[i], tags=tags[i]) for i in range(len(words))] ##CHECK
            #senten = sent_tokenize(paragraph)
#            for sent in senten:
#                sent = word_stem(sent)
#                tok_corpus.append(word_tokenize(sent))
        #tok_corpus+=[TaggedDocument(words=word_tokenize(word_stem(sent)), tags=[str(i)]) for i, _d in enumerate(paragraph) for sent in sent_tokenize(_d)] ##CHECK
#        print(tok_corpus)
        # building vocab
        self.d2v = Doc2Vec(dm=0, vector_size=300, min_count=5, negative=5, hs=0, sample=0, epochs=40) #it was 2, but it says that it works better with min_count=5
        self.d2v.build_vocab(tok_corpus)
        self.d2v.train(tok_corpus, total_examples=self.d2v.corpus_count, epochs=self.d2v.epochs)
        
    def __build_paragraph_embeddings(self):
        vectors = []
        for element in self.corpus['text']:
            element_tokens = []
            senten = sent_tokenize(element)
            for sent in senten:
                sent = word_stem(sent)
                element_tokens += word_tokenize(sent)
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

if __name__ == '__main__':
    article_paths = CovidDataLoader.load_articles_paths(data_root_path)

    abstracts = CovidDataLoader.load_data(article_paths, offset=0, limit=10000, load_sentences=False, preprocess=False)
    query1 = word_stem("Main risk factors of covid19")
    query2 = word_stem("Does smoking increase risks when having covid19?")
    query3 = word_stem("What is the mortality rate of covid19?")

    query_engine = D2VQueryEngine()
    query_engine.fit(abstracts)
    results = query_engine.run_query(query1)