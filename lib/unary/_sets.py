

import typing

from lib.unary import _sized
from lib.unary.main import unary




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


def to_set_union(
    iterable:typing.Iterable):

    @unary()
    def get_set_union(
        set_items:typing.Set):

        return set_items.union(iterable)

    return get_set_union


def to_set_intersection(
    iterable:typing.Iterable):

    @unary()
    def get_set_intersection(
        set_items:typing.Set):

        return set_items.intersection(iterable)

    return get_set_intersection


def to_set_difference(
    iterable:typing.Iterable):

    @unary()
    def get_set_difference(
        set_items:typing.Set):

        return set_items.difference(iterable)

    return get_set_difference


def to_bool_is_disjoint_from(
    set_other:typing.Set):

    return to_set_intersection(set_other) \
        | _sized.to_bool_is_empty()


# TODO implement: create negated version
def to_bool_does_contain(
    item:typing.Any):

    @unary()
    def inner(
        set_items:typing.Set):

        return item in set_items

    return inner

