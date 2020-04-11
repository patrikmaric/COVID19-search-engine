from nltk import sent_tokenize

from settings import data_root_path

import pandas as pd

def extract_data_from_dict(d1, keys, mandatory_keys=()):
    """
    Given a dictionary keeps only data defined by the keys param

    Args:
        d1: dictionary to extract the data from
        keys: keys to extract
        mandatory_keys: if any of this keys isn't in the d1, exception will be risen

    Returns:
        dictionary containing data from d1 defined by keys param
    """
    for k in mandatory_keys:
        if k not in d1:
            raise ValueError('Not all mandatory keys are in the given dictionary')
    d2 = {k: d1[k] for k in keys if k in d1}
    return d2


def load_sentences_from_abstracts(abstracts):
    sentences = []
    for a in abstracts:
        sents = sent_tokenize(a['text'])

        for i in range(len(sents)):
            sent = {k: v for k, v in a.items()}
            sent['text'] = sents[i]
            sent['position'] = i
            sentences.append(sent)
    return pd.DataFrame(sentences)

