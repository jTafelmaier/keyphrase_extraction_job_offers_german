

import typing

from lib.unary import _items
from lib.unary import _sequences
from lib.unary import _sized
from lib.unary.main import unary
from lib.unary.main import Function_unary




def to_int_length():
    return _sized.to_int_length()


def to_bool_is_empty():
    return _sized.to_bool_is_empty()


def to_bool_is_not_empty():
    return _sized.to_bool_is_not_empty()


def to_bool_is_singleton():
    return _sized.to_bool_is_singleton()


def to_bool_is_larger_than_singleton():
    return _sized.to_bool_is_larger_than_singleton()


def to_item_at_index(
    int_index:int):

    return _sequences.to_item_at_index(int_index)


def to_item_first():
    return _sequences.to_item_first()


def to_item_last():
    return _sequences.to_item_last()


# TODO create a separate file "_pairs.py" and move this function there, since it is only meaningful on 2-tuples and not tuples in general
def to_pair_mapped(
    function_first:Function_unary = _items.to_item_unmodified(),
    function_last:Function_unary = _items.to_item_unmodified()):

    def get_bool_is_pure():

        return function_first.bool_is_pure \
            and function_last.bool_is_pure

    @unary(bool_is_pure=get_bool_is_pure())
    def get_pair_mapped(
        pair:typing.Tuple):

        return (
            pair \
                >> to_item_first() \
                >> function_first,
            pair \
                >> to_item_last() \
                >> function_last)

    return get_pair_mapped

