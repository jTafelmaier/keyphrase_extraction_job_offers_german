

import typing

from lib.unary import _sequences
from lib.unary import _sized
from lib.unary.main import unary_pure




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


# TODO refactor: perhaps rename to "to_list_trimmed"
# TODO refactor: perhaps move to _sized.py
def to_list_slice(
    index_start:typing.Optional[int] = None,
    index_end:typing.Optional[int] = None,
    int_step:typing.Optional[int] = None):

    @unary_pure()
    def get_list_slice(
        list_items:typing.List):

        return list_items[
            index_start:
            index_end:
            int_step]

    return get_list_slice


def to_list_replicated(
    number_replications:int):

    @unary_pure()
    def get_list_replicated(
        list_items:typing.List):

        return list_items \
            * number_replications

    return get_list_replicated


def to_iterable_pairs_of_neighbors():

    @unary_pure()
    def get_iterable_pairs_of_neighbors(
        list_items:typing.List):

        return zip(
            list_items[:-1],
            list_items[1:])

    return get_iterable_pairs_of_neighbors

