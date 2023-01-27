

import typing

from lib.unary.main import unary_pure




def to_item_at_index(
    int_index:int):

    @unary_pure()
    def get_item(
        sequence_items:typing.Sequence):

        return sequence_items.__getitem__(int_index)

    return get_item


def to_item_first():

    return to_item_at_index(0)


def to_item_last():

    return to_item_at_index(-1)

