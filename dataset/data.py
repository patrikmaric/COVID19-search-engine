import json
import tqdm
import pandas as pd

from pathlib import Path

from nltk import sent_tokenize

from dataset.util import extract_data_from_dict
from settings import data_root_path

abstract_keys = ('section', 'text')


class CovidDataLoader():

    @staticmethod
    def load_articles_paths(root_path=data_root_path, file_extension='json'):
        """
        Gets the paths to all files with the given file extension,
        in the given directory(root_path) and all its subdirectories.

        Args:
            root_path: path to directory to get the files from
            file_extension: extension to look for

        Returns:
            list of paths to all articles from the root directory
        """
        article_paths = []
        for path in Path(root_path).rglob('*.%s' % file_extension):
            article_paths.append(str(path))
        return article_paths

    @staticmethod
    def load_abstracts_data(articles_paths, offset=0, limit=None, keys=abstract_keys, load_sentences=False):
        """
        Given the list of paths to articles json files, returns pandas DataFrame or list of
        dictionaries(depends on the reutrn_df parameter) containing the info defined by the
        given keys.

        Args:
            articles_paths: list of paths to articles to load
            offset: loading start index in the articles_paths list
            limit: number of articles to load
            return_df: if true returns pandas DataFrame, otherwise list of dictionaries
            keys: keys for abstracts in the covid dataset_ to include in the loaded data

        Returns:
             abstracts data as a pandas DataFrame or a list of dictionaries (defined via return_df parameter)
        """
        N = len(articles_paths)
        assert offset < N
        last_index = N
        if limit and offset + limit < N:
            last_index = offset + limit

        abstracts = []
        for path in tqdm.tqdm(articles_paths[offset:last_index]):
            with open(path, 'r') as f:
                curr_article = json.load(f)
                if 'abstract' in curr_article:
                    for section in curr_article['abstract']:
                        curr_abstract_part = {'paper_id': curr_article['paper_id']}
                        try:
                            curr_abstract_part.update(extract_data_from_dict(section, keys, mandatory_keys=['text']))
                            abstracts.append(curr_abstract_part)
                        except:
                            pass

        if load_sentences:
            return CovidDataLoader.__load_sentences(abstracts)
        return abstracts

    @staticmethod
    def __load_sentences(abstracts):
        sentences = []
        for a in abstracts:
            sents = sent_tokenize(a['text'])

            for i in range(len(sents)):
                sent = {k: v for k, v in a.items()}
                sent['text'] = sents[i]
                sent['position'] = i
                sentences.append(sent)
        return pd.DataFrame(sentences)


if __name__ == '__main__':
    article_paths = CovidDataLoader.load_articles_paths(data_root_path)
    abstracts = CovidDataLoader.load_abstracts_data(article_paths, offset=5000, limit=10, load_sentences=True)

