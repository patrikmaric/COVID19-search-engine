from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec, Doc2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing.preprocessing import preprocess_data
from preprocessing.preprocessing import word_stem

import pandas as pd

import numpy as np

# from tqdm.notebook import tqdm

# Enters paragraph; make sentences and words to feed W2V
#from data import CovidDataLoader
from dataset.data import CovidDataLoader
from query_model.queries import QueryEngine


# train_corpus = whole corpus tokenized
from settings import data_root_path


def doc2vector(corpus, query):
    #    model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
    #    model.build_vocab(train_corpus)
    #    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    #    give paragraph as list of tokens to get vector:
    #    vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
    pass


class W2VQueryEngine(QueryEngine):

    def __init__(self):
        super().__init__()

    def fit(self, corpus):
        self.corpus = corpus  # expect pd.DataFrame
        self.__build_w2v()
        self.__build_paragraph_embeddings()

    def run_query(self, query, n=5):
        # TODO; check this
        query_vector = self.get_paragraph_embedding(query).reshape(1, -1)
        similarities = cosine_similarity(self.paragraph_vectors,
                                         query_vector)  # TODO: check if this already sorts values
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
            vectors.append(element_vec)

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

if __name__ == '__main__':
    article_paths = CovidDataLoader.load_articles_paths(data_root_path)
    abstracts = CovidDataLoader.load_data(article_paths, offset=5000, limit=10, load_sentences=False, preprocess=False)
    # TODO: input
    #abstracts_text = list(abstracts['text'])
    query1 = word_stem("Main risk factors for covid19")
    query2 = word_stem("Does smoking increase risks when having covid19?")
    query3 = word_stem("What is the mortality rate of covid19?")

    query_engine = W2VQueryEngine()
    query_engine.fit(abstracts)


    """model = build_model(abstracts_text)  # needs to send whole paragraphs
    print("Main risk factors for covid19")
    word2vector(abstracts_text, query1, model)
    print("Does smoking increase risks when having covid19?")
    word2vector(abstracts_text, query2, model)
    print("What is the mortality rate of covid19?")
    word2vector(abstracts_text, query3, model)"""