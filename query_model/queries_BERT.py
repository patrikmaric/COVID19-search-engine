from settings import data_root_path
from dataset.data import CovidDataLoader

from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from nltk import sent_tokenize

import pickle
import pandas as pd
import numpy as np
import time


def normalize(embeddings):
        """
        Normalizes embeddings using L2 normalization.
        Args:
            embeddings: input embeddings matrix
        Returns:
            normalized embeddings
        """
        # Calculation is different for matrices vs vectors
        if len(embeddings.shape) > 1:
            return embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1)

        else:
            return embeddings / np.linalg.norm(embeddings)
        

def bioBERT_embeddings(data, query=False):
    
    """
    Input:
        corpus: list of dictionaries, each containing information about one paragraph: paper_id, section, text
        query: if True, import is one sentence - a query
    Returns:
        corpus embeddings: numpy array containing paragraph embeddings for each text paragraph in input
        which is obtained by averaging over sentence embeddings(try #1 - until a better idea arrives (probably not so great))
        -dimensions: n x 768 where n represents number of input paragraphs
    
    References
    ----------
    {
    reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
    }

    """
    
    ## Use bioBERT for mapping tokens to embeddings
    ## Insert your own path to model
    
    #word_embedding_model = models.Transformer('C:/Users/marko/Desktop/Marko/biobert/')

    # Apply mean/max pooling to get one fixed sized sentence vector
    #pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   #pooling_mode_mean_tokens=True,
                                   #pooling_mode_cls_token=False,
                                   #pooling_mode_max_tokens=False)

    #model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    model = SentenceTransformer('bert-base-nli-mean-tokens') #najbrži od mogućih(po durationu na 3 rečenice)
    
    if query:
        return normalize(np.array(model.encode([data])).reshape(1,768))
    
    else:
        text_paragraphs = [paragraph for paragraph in data['text']]
        n=len(text_paragraphs)
        
        corpus_embeddings=[]
        for paragraph in text_paragraphs:
            sentences = sent_tokenize(paragraph)
            sent_embeddings = normalize(np.array(model.encode(sentences)).reshape(-1,768))#shape = no_of_sents_in_paragraph X 768
            corpus_embeddings.append(np.mean(sent_embeddings,axis=0).reshape(1,768)) 
        
        return normalize(np.array(corpus_embeddings).reshape(n,768))





class QueryEngine_BERT():

    def __init__(self):
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
        self.corpus = [paragraph for paragraph in abstracts['text']]
        self.corpus_embeddings = bioBERT_embeddings(corpus, query=False)
        
    def __create_query_result(self, query, similarities, n):
        """

        Args:
            similarities: sparse matrix containing cosine similarities between the query vector and documents from corpus
            n: number of most similar documents to include in the result

        Returns:
            pandas DataFrame containing query, document, similarity
        """

        result = {
            'query': query * len(similarities),
            'text': self.corpus,
            'sim': np.squeeze(similarities)
        }
        if self.ids:
            result.update({'id': self.ids})

        result = pd.DataFrame(result).sort_values(by='sim', ascending=False)[:n]

        return result[result['sim'] > 0]

    def run_query(self, query, n=5):
        """
        Runs the given query, returns max n most similar documents from the corpus on which the model was build.

        Args:
            query: query to run
            n: max number of results returned

        Returns:
            n(or less) most similar documents from the corpus on which the model was build
        """
        if self.corpus is None:
            raise AttributeError('Model not built yet, please call the fit method before running queries!')

        assert type(query) == str
            

        query_embedding = bioBERT_embeddings(query, query=True)
        similarities = np.dot(self.corpus_embeddings,query_embedding.T)  # TODO: check if this already sorts values
        
        return self.__create_query_result(query, similarities, n)

    def save(self, dir_path, name):
        """
        Serializes the object to file(name.dat) to the directory defined by the path.

        Args:
            dir_path: path of the directory to save the object to
            name: name of the file without any extensions
        """
        pickle_path = dir_path + name + '.dat'
        print('Writing object to %s' % pickle_path)
        with open(pickle_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(pickle_path):
        """
        Loads(de-serializes) QueryEngine object from the given path.

        Args:
            pickle_path: path to QueryEngine pickle

        Returns:
            QueryEngine object
        """
        with open(pickle_path, 'rb') as f:
            query_engine = pickle.load(f)
            if type(query_engine) != QueryEngine_BERT:
                raise ValueError('Path to non QueryEngine_BERT object!')
            return query_engine


if __name__ == '__main__':
    
    
    article_paths = CovidDataLoader.load_articles_paths(data_root_path)
    #abstracts = CovidDataLoader.load_abstracts_data(article_paths, offset=5000, limit=1000, load_sentences=False)
    abstracts = CovidDataLoader.load_data(article_paths, offset=10000, limit=1000, load_sentences=False, preprocess=False)
    
    paper_ids = [paper_id for paper_id in abstracts['paper_id']]

    
    tic=time.time()
    query_engine = QueryEngine_BERT()
    query_engine.fit(abstracts, paper_ids)
    
    query_engine.save('./', 'abstracts_query_engine')
    toc=time.time()
    duration=toc-tic
    

