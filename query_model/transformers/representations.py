from nltk.tokenize import sent_tokenize, word_tokenize 
from gensim.models import Word2Vec, Doc2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing.preprocessing import preprocess_data
from preprocessing.preprocessing import word_stem
#from tqdm.notebook import tqdm

#Enters paragraph; make sentences and words to feed W2V
def word2vector(corpus,query,model):
    par = ""
    result = 0.0
    query_vec = get_paragraph_embedding(model,query).reshape(1,-1)
    for paragraph in corpus:
        paragraph_vec = get_paragraph_embedding(model, paragraph).reshape(1,-1)
        cos_lib = cos_similarity(paragraph_vec, query_vec)
        if cos_lib > result:
            result = cos_lib
            par = paragraph
    print('par',par)
    print('res',result)


def fit(corpus):
        pass
    
    
def cos_similarity(par_vec, query_vec):
        return cosine_similarity(par_vec, query_vec)[0][0]
    
def build_model(corpus):
    tok_corpus = []
    for paragraph in corpus:
        senten = sent_tokenize(paragraph)
        for sent in senten:
            sent = word_stem(sent)
            tok_corpus.append(word_tokenize(sent))
    #building vocab
    model = Word2Vec(tok_corpus,min_count=1,size=50,workers=3,window=5,sg=0)
    return model

##Add up word2vec
def get_paragraph_embedding(model,paragraph):
    sentences = sent_tokenize(paragraph)
    word_list = []
    for sent in sentences:
        sent = word_stem(sent)
        word_list.append(word_tokenize(sent))
    words = []
    for sublist in word_list:
        for item in sublist:
            words.append(item)
    result_vec = np.zeros(np.shape(model[list(model.wv.vocab.keys())[0]]))
    for word in words: 
        if word in model.wv.vocab.keys():
            result_vec += np.array(model[word])
    return result_vec


#train_corpus = whole corpus tokenized
def doc2vector(corpus,query):
#    model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
#    model.build_vocab(train_corpus)
#    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
#    give paragraph as list of tokens to get vector:
#    vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
    pass
