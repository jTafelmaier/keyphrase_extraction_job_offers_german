

import typing

from pke import unsupervised as _pke_unsupervised
import spacy

from lib.unary import _iters
from lib.unary import _tuples
from lib.unary.bs4 import _htmls
from lib.unary.main import unary




def get_list_lists_texts_keywords_lowercase_load_model(
    list_htmls:typing.List[str]):

    model = _pke_unsupervised.TextRank()

    spacy_model = spacy.load("de_dep_news_trf")

    @unary()
    def get_list_texts_keywords_local(
        html:str):

        text = html \
            >> _htmls.to_text_content()

        model \
            .load_document(
                input=text,
                language="ge",
                spacy_model=spacy_model)

        model \
            .candidate_selection()

        model \
            .candidate_weighting()

        return model \
            .get_n_best(
                n=15) \
            >> _iters.to_iterable_mapped(_tuples.to_item_first()) \
            >> _iters.to_list()

    return list_htmls \
        >> _iters.to_iterable_mapped(get_list_texts_keywords_local) \
        >> _iters.to_list()

