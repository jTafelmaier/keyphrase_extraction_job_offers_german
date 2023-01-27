

import typing

from lib.unary import _numbers
from lib.unary.main import unary_pure




def to_int_length():

    # TODO investigate if always pure
    @unary_pure()
    def get_length(
        object_sized:typing.Sized):

        return object_sized.__len__()

    return get_length


def to_bool_is_empty():
    return to_int_length() \
        | _numbers.to_bool_is_lower_than(1)


def to_bool_is_not_empty():
    return to_int_length() \
        | _numbers.to_bool_is_higher_than(0)


def to_bool_is_singleton():
    return to_int_length() \
        | _numbers.to_bool_is_equal_to(1)


def to_bool_is_larger_than_singleton():
    return to_int_length() \
        | _numbers.to_bool_is_higher_than(1)

