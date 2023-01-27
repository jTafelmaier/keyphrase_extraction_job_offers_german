

import bs4

from lib.unary.bs4 import _tags
from lib.unary.main import unary_pure




ALIAS_BS4_PARSER_HTML = "html.parser"


def to_soup():

    @unary_pure()
    def inner(
        html:str):

        return bs4.BeautifulSoup(
            markup=html,
            features=ALIAS_BS4_PARSER_HTML)

    return inner


# TODO test further
def to_text_content(
    text_separator:str = "\n",
    do_strip_invisible_characters:bool = True):

    @unary_pure()
    def inner(
        html:str):

        return html \
            >> to_soup() \
            >> _tags.to_text_content(
                text_separator=text_separator,
                do_strip_invisible_characters=do_strip_invisible_characters)

    return inner

