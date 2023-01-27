

import requests

from lib.unary.main import unary




# TODO refactor: move
def to_text_response():

    @unary()
    def inner(
        response:requests.Response):

        return response.text

    return inner

