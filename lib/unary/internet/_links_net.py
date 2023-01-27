

import typing

from lib.unary import _jsons
from lib.unary.bs4 import _htmls
from lib.unary.requests import _links
from lib.unary.requests import _responses
from lib.unary.main import unary




def to_soup_get(
    dict_headers:typing.Dict[str, str] = {"User-Agent": "Magic Browser"},
    do_use_ssl_verification:bool = True):

    @unary()
    def inner(
        link:str):

        return link \
            >> _links.to_response_get(
                dict_headers=dict_headers,
                do_use_ssl_verification=do_use_ssl_verification) \
            >> _responses.to_text_response() \
            >> _htmls.to_soup()

    return inner


def to_soup_post(
    text_data:str,
    dict_headers:typing.Dict[str, str] = {"User-Agent": "Magic Browser"},
    do_use_ssl_verification:bool = True):

    @unary()
    def inner(
        link:str):

        return link \
            >> _links.to_response_post(
                text_data=text_data,
                dict_headers=dict_headers,
                do_use_ssl_verification=do_use_ssl_verification) \
            >> _responses.to_text_response() \
            >> _htmls.to_soup()

    return inner


# TODO refactor: rename: does not always return dict
def to_dict_json_get(
    dict_headers:typing.Dict[str, str] = {"User-Agent": "Magic Browser"},
    do_use_ssl_verification:bool = True):

    @unary(bool_is_pure=False)
    def inner(
        link:str):

        return link \
            >> _links.to_response_get(
                dict_headers=dict_headers,
                do_use_ssl_verification=do_use_ssl_verification) \
            >> _responses.to_text_response() \
            >> _jsons.to_dict_from_json()

    return inner


# TODO refactor: rename: does not always return dict
def to_dict_post(
    text_data:str = None,
    dict_json:typing.Dict = None,
    dict_headers:typing.Dict[str, str] = {"User-Agent": "Magic Browser"}):

    @unary()
    def inner(
        link:str):

        return link \
            >> _links.to_response_post(
                text_data=text_data,
                dict_json=dict_json,
                dict_headers=dict_headers) \
            >> _responses.to_text_response() \
            >> _jsons.to_dict_from_json()

    return inner

