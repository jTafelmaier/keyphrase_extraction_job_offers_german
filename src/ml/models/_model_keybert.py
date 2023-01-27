

import typing

import keyphrase_vectorizers as _module_keyphrase_vectorizers
import keybert as _module_keybert
from flair import embeddings as _flair_embeddings

from lib.unary import _iters
from lib.unary import _sequences
from lib.unary.bs4 import _htmls
from lib.unary.main import unary




# NOTE using parts from "https://towardsdatascience.com/enhancing-keybert-keyword-extraction-results-with-keyphrasevectorizers-3796fa93f4db"
def get_list_lists_texts_keywords_load_model(
    list_htmls:typing.List[str]):

    model = _module_keybert.KeyBERT(
            model=_flair_embeddings.TransformerDocumentEmbeddings("dbmdz/bert-base-german-uncased"))

    vectorizer = _module_keyphrase_vectorizers.KeyphraseCountVectorizer(
            spacy_pipeline="de_dep_news_trf",
            stop_words="german",
            lowercase=False)

    @unary()
    def get_list_texts_keywords(
        html:str):

        text = html \
            >> _htmls.to_text_content()

        try:
            # NOTE multiple documents are not passed here since the model sometimes raises an IndexError (for unknown reasons), causing the entire computation to fail.
            return model \
                .extract_keywords(
                    docs=text,
                    vectorizer=vectorizer,
                    top_n=15) \
                >> _iters.to_iterable_mapped(_sequences.to_item_first()) \
                >> _iters.to_list()

        except IndexError:
            return []

    return list_htmls \
        >> _iters.to_iterable_mapped(get_list_texts_keywords) \
        >> _iters.to_list()

