

import webbrowser

from lib.unary.main import unary




# TODO test further
def to_bool_open_link_new_tab():

    @unary()
    def inner(
        link:str):

        return webbrowser.open_new_tab(link)

    return inner

