

import typing

from sklearn.feature_extraction import text as _sk_fe_text

from lib.unary import _items
from lib.unary import _iters
from lib.unary import _tuples
from lib.unary.bs4 import _htmls
from lib.unary.main import unary

from src.ml import _ml_retrieval




def get_model_new_trained():

    dataframe_data_training = _ml_retrieval.get_dataframe_data_partition(_ml_retrieval.PATH_CSV_DATA_TRAINING)

    list_texts_content_of_html = _ml_retrieval.get_list_texts_content_of_html(dataframe_data_training)

    model = _sk_fe_text.TfidfVectorizer()

    model \
        .fit(list_texts_content_of_html)

    return model


def get_list_texts_keywords_lowercase(
    html:str,
    model:_sk_fe_text.TfidfVectorizer):

    list_text_html = html \
        >> _htmls.to_text_content() \
        >> _items.to_list_singleton()

    list_names_features = model \
        .get_feature_names_out() \
        .tolist()

    # TODO implement: perhaps use threshold
    return model \
        .transform(list_text_html) \
        .toarray() \
        .ravel()  \
        .tolist() \
        >> _iters.to_iterable_zipped(list_names_features) \
        >> _iters.to_list_sorted(
            function_get_key=_tuples.to_item_first(),
            do_sort_descending=True) \
        >> _iters.to_iterable_limit_count(15) \
        >> _iters.to_iterable_mapped(_tuples.to_item_last()) \
        >> _iters.to_list()


def get_list_lists_texts_keywords_lowercase_load_model(
    list_htmls:typing.List[str]):

    model = get_model_new_trained()

    @unary()
    def get_list_texts_keywords_lowercase_local(
        html:str):

        return get_list_texts_keywords_lowercase(
                html=html,
                model=model)

    return list_htmls \
        >> _iters.to_iterable_mapped(get_list_texts_keywords_lowercase_local) \
        >> _iters.to_list()

