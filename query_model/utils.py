import numpy as np
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer

# TODO: remove possible duplicates in corpus and extremly short paragraphs

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


def BERT_sentence_embeddings(data, text_column=None, query=False):
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

    # pre-trained model on semantic text similarity task
    model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

    if query:
        return normalize(np.array(model.encode([data])).reshape(1, 768))

    elif text_column:
        text_paragraphs = [paragraph for paragraph in list(data[text_column])]
        n = len(text_paragraphs)

        corpus_embeddings = []
        for paragraph in text_paragraphs:
            sentences = sent_tokenize(paragraph)
            sent_embeddings = normalize(
                np.array(model.encode(sentences)).reshape(-1, 768))  # shape = no_of_sents_in_paragraph X 768
            corpus_embeddings.append(np.mean(sent_embeddings, axis=0).reshape(1, 768))

        return normalize(np.array(corpus_embeddings).reshape(n, 768))

    raise AttributeError('Input must be either a query, or training data!')