import numpy as np
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer

from dataset.data import CovidDataLoader
from query_model.queries import QueryEngine
from settings import data_root_path

abstract_keys = ('section', 'text')
body_text_keys = ('section', 'text')

#TODO: remove possible duplicates in corpus and extremly short paragraphs

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
        

def BERT_sentence_embeddings(data, query=False):
    
    """
    Input:
        corpus: DataFrame containing information about paragraphs : paper_id, section, text
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
    
    #pre-trained model on semantic text similarity task
    model = SentenceTransformer('bert-base-nli-stsb-mean-tokens') 
    
    if query:
        return normalize(np.array(model.encode([data])).reshape(1,768))
    
    else:
        text_paragraphs = [paragraph for paragraph in list(data['text'])]
        n=len(text_paragraphs)
        
        corpus_embeddings=[]
        for paragraph in text_paragraphs:
            sentences = sent_tokenize(paragraph)
            sent_embeddings = normalize(np.array(model.encode(sentences)).reshape(-1,768))#shape = no_of_sents_in_paragraph X 768
            corpus_embeddings.append(np.mean(sent_embeddings,axis=0).reshape(1,768)) 
        
        return normalize(np.array(corpus_embeddings).reshape(n,768))


class BERTQueryEngine(QueryEngine):

    def __init__(self):
        super().__init__()

    def fit(self, corpus, document_ids=None):
        self.ids = document_ids
        self.corpus = [paragraph for paragraph in corpus['text']]
        self.corpus_embeddings = BERT_sentence_embeddings(corpus, query=False)

    def run_query(self, query, n=5):
        if self.corpus is None:
            raise AttributeError('Model not built yet, please call the fit method before running queries!')

        assert type(query) == str
            

        query_embedding = BERT_sentence_embeddings(query, query=True)
        similarities = np.dot(self.corpus_embeddings,query_embedding.T)  # TODO: check if this already sorts values
        
        return self.__create_query_result(query, similarities, n)


if __name__ == '__main__':
    
    # preferably run on GPU, sentence transformers model encodes up to 1300 sentences/s
    # much slower on CPU, between 30 and 50 sentences/s
    
    article_paths = CovidDataLoader.load_articles_paths(data_root_path)
    abstracts = CovidDataLoader.load_data(article_paths, offset=0, limit=None, load_sentences=False, preprocess=False)
    body_texts = CovidDataLoader.load_data(article_paths, key='body_text', keys=body_text_keys, offset=0, limit=None, load_sentences=False, preprocess=False)
    paper_ids = [paper_id for paper_id in body_texts['paper_id']]
    
    
    #query_engine = BERTQueryEngine()
    #query_engine.fit(body_texts, paper_ids)
    
    #query_engine.save('./', 'body_texts_query_engine')