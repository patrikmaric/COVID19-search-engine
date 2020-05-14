from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
from preprocessing.preprocessing import preprocess_query

import pandas as pd

#from data import CovidDataLoader
from dataset.data import CovidDataLoader
from query_model.queries import QueryEngine

from settings import data_root_path


#requires preprocessed text; shouldn't load sentences
#INPUT: pd.Dataframe - abstracts; query - string
class D2VQueryEngine(QueryEngine):

    def __init__(self):
        super().__init__()

    def fit(self, corpus):
        self.corpus = corpus
        self.__build_d2v()
        self.__build_paragraph_embeddings()

    def run_query(self, query, n=5, q=True):
        query = preprocess_query(query, q)[0]
        query_tokens = []
        senten = sent_tokenize(query)
        for sent in senten:
            tok = word_tokenize(sent)
            tok.remove('.')
            query_tokens += tok
        query_vector = self.d2v.infer_vector(query_tokens).reshape(1,-1)
        n1 = np.linalg.norm(query_vector)
        qvn = np.divide(query_vector, n1)
        n2 = np.linalg.norm(self.paragraph_vectors, axis=1)
        pvn = np.divide(self.paragraph_vectors, n2.reshape(-1, 1))

        similarities = qvn.dot(pvn.T)[0]
        return self.__create_query_result(query, similarities, n)

    def __build_d2v(self):
        tok_corpus = []
        cnt = 0
        for paragraph in self.corpus['preprocessed_text']:
            cnt += 1
            j = str(cnt)
            sentn = sent_tokenize(paragraph)
            par_words = []
            tags = []
            for i,sent in enumerate(sentn):  
                tags.append(j + str(i))
                par_words.append(word_tokenize(sent))
                for element in par_words:
                    if '.' in element:
                        element.remove('.')
            tok_corpus+=[TaggedDocument(words=par_words[j], tags=tags[j]) for j in range(len(par_words))]
        # building vocab
        self.d2v = Doc2Vec(dm=0, vector_size=300, min_count=5, negative=5, hs=0, sample=0, epochs=400, window=15) #it was 2, but it says that it works better with min_count=5
        self.d2v.build_vocab(tok_corpus)
        self.d2v.train(tok_corpus, total_examples=self.d2v.corpus_count, epochs=self.d2v.epochs)
        
    def __build_paragraph_embeddings(self):
        vectors = []
        for element in self.corpus['preprocessed_text']:
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

if __name__ == '__main__':
    article_paths = CovidDataLoader.load_articles_paths(data_root_path)
    abstracts = CovidDataLoader.load_data(article_paths,key='abstract', offset=0, limit=100, load_sentences=False, preprocess=True, q=False)
    print(abstracts)
    query1 = "Main risk factors of covid19"
#    query2 = word_stem("Does smoking increase risks when having covid19?")
#    query3 = word_stem("What is the mortality rate of covid19?")

    query_engine = D2VQueryEngine()
    
    query_engine.fit(abstracts)
    results = query_engine.run_query(query1)