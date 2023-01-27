

import typing

import requests

from lib.unary.main import unary




# TODO test further
def to_response_get(
    dict_headers:typing.Dict = {"User-Agent": "Magic Browser"},
    do_use_ssl_verification:bool = True):

    @unary()
    def inner(
        link:str):

        return requests.get(
            url=link,
            headers=dict_headers,
            verify=do_use_ssl_verification)

    return inner


# TODO test further
def to_response_post(
    text_data:str = None,
    dict_json:typing.Dict = None,
    dict_headers:typing.Dict[str, str] = {"User-Agent": "Magic Browser"},
    do_use_ssl_verification:bool = True):

    @unary()
    def inner(
        link:str):

        return requests.post(
            url=link,
            data=text_data,
            json=dict_json,
            headers=dict_headers,
            verify=do_use_ssl_verification)

    return inner

