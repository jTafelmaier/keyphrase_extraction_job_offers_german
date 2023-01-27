

import typing

from lib.unary import _iters
from lib.unary import _tuples
from lib.unary.main import Function_unary
from lib.unary.main import unary
from lib.unary.main import unary_pure




# TODO test further
def to_iterable_values():

    @unary()
    def get_values(
        dictionary:typing.Dict):

        return dictionary \
            .values()

    return get_values


# TODO test further
def to_iterable_keys():

    @unary()
    def get_keys(
        dictionary:typing.Dict):

        return dictionary \
            .keys()

    return get_keys


def to_iterable_pairs():

    @unary()
    def get_pairs(
        dictionary:typing.Dict):

        return dictionary \
            .items()

    return get_pairs


def to_item_get(
    item_key:typing.Any,
    do_allow_default:bool = True,
    item_default:typing.Any = None):

    @unary()
    def get_value(
        dictionary:typing.Dict):

        if do_allow_default:
            return dictionary \
                .get(
                    item_key,
                    item_default)
        else:
            return dictionary \
                [item_key]

    return get_value


def to_item_at(
    item_key:typing.Any):

    @unary()
    def inner(
        dictionary:typing.Dict):

        return dictionary[item_key]

    return inner


# TODO test further
def to_dict_inplace_set(
    item_key:typing.Hashable,
    item_value:typing.Any):

    @unary()
    def inner(
        dictionary:typing.Dict):

        dictionary[item_key] = item_value
        return dictionary

    return inner


def to_dict_inplace_remove(
    item_key:typing.Any,
    do_allow_default:bool = True,
    item_default:typing.Any = None):

    @unary()
    def get_value_remove_key(
        dictionary:typing.Dict):

        if do_allow_default:
            return dictionary \
                .pop(
                    item_key,
                    item_default)
        else:
            return dictionary \
                .pop(item_key)

    return get_value_remove_key


def to_dict_map_values(
    function:Function_unary):

    @unary_pure()
    def get_dict_map_values(
        dictionary:typing.Dict):

        return dictionary \
            >> to_iterable_pairs() \
            >> _iters.to_iterable_mapped(_tuples.to_pair_mapped(
                    function_last=function)) \
            >> _iters.to_dict()

    return get_dict_map_values


def to_dict_inplace_update( # TODO create version that updates inplace when reference counter is 1 (current function)
    dict_other:typing.Dict):

    @unary(bool_is_pure=False)
    def get_dict_updated(
        dictionary:typing.Dict):

        dictionary.update(dict_other)
        return dictionary

    return get_dict_updated

