

from lib.unary.main import unary_pure




# TODO test further
def to_text():

    @unary_pure()
    def inner(
        integer:int):

        return integer.__str__()

    return inner


# TODO test further
def to_iterable_from_0(
    int_step:int = 1):

    @unary_pure()
    def inner(
        integer:int):

        return range(
            0,
            integer,
            int_step)

    return inner

