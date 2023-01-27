

from lib.unary.main import unary




def to_bool_not():

    @unary()
    def get_negated(
        boolean:bool):

        return not boolean

    return get_negated


def to_bool_and(
    bool_other:bool):

    @unary()
    def get_bool_and(
        boolean:bool):

        return boolean and bool_other

    return get_bool_and


# TODO test further
def to_bool_or(
    bool_other:bool):

    @unary()
    def inner(
        boolean:bool):

        return boolean or bool_other

    return inner

