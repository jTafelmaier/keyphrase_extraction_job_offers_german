

import typing

import yake

from lib.unary import _iters
from lib.unary import _tuples
from lib.unary.bs4 import _htmls
from lib.unary.main import unary




def get_model_new_trained():

    return yake.KeywordExtractor(
            lan="de",
            n=1,
            top=15)


def get_list_texts_keywords(
    html:str,
    model:yake.KeywordExtractor):

    text_html = html \
        >> _htmls.to_text_content()

    return model \
        .extract_keywords(text_html) \
        >> _iters.to_iterable_mapped(_tuples.to_item_first()) \
        >> _iters.to_list()


def get_list_lists_texts_keywords_load_model(
    list_htmls:typing.List[str]):

    model = get_model_new_trained()

    @unary()
    def get_list_texts_keywords_local(
        html:str):

        return get_list_texts_keywords(
                html=html,
                model=model)

    return list_htmls \
        >> _iters.to_iterable_mapped(get_list_texts_keywords_local) \
        >> _iters.to_list()

