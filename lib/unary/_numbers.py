

import typing

from lib.unary.main import unary
from lib.unary.main import unary_pure




def to_bool_is_equal_to(
    number_other:typing.Union[int, float]):

    @unary()
    def get_bool_is_equal_to(
        number:typing.Union[int, float]):

        return number == number_other

    return get_bool_is_equal_to


# TODO test
def to_bool_is_not_equal_to(
    number_other:typing.Union[int, float]):

    @unary()
    def get_bool_is_not_equal_to(
        number:typing.Union[int, float]):

        return number != number_other

    return get_bool_is_not_equal_to


def to_bool_is_lower_than(
    number_other:typing.Union[int, float]):

    @unary()
    def get_bool_is_lower_than(
        number_x:typing.Union[int, float]):

        return number_x < number_other

    return get_bool_is_lower_than


def to_bool_is_higher_than(
    number_other:typing.Union[int, float]):

    @unary()
    def get_bool_is_higher_than(
        number_x:typing.Union[int, float]):

        return number_x > number_other

    return get_bool_is_higher_than


# TODO test further
def to_int_floored():

    @unary_pure()
    def get_int_floored(
        number_x:typing.Union[int, float]):

        return number_x.__floor__()

    return get_int_floored


def to_int_ceiling():

    @unary_pure()
    def inner(
        number_x:typing.Union[int, float]):

        return number_x.__ceil__()

    return inner


def to_number_added_by(
    number_other:typing.Union[int, float]):

    @unary_pure()
    def inner(
        number_x:typing.Union[int, float]):

        return number_x + number_other

    return inner


def to_number_subtracted_by(
    number_other:typing.Union[int, float]):

    @unary_pure()
    def inner(
        number_x:typing.Union[int, float]):

        return number_x - number_other

    return inner


def to_number_multiplied_by(
    number_other:typing.Union[int, float]):

    @unary_pure()
    def get_number_multiplied(
        number_x:typing.Union[int, float]):

        return number_x * number_other

    return get_number_multiplied


def to_int_multiplied_by(
    number_other:typing.Union[int, float]):

    @unary_pure()
    def get_int_multiplied(
        number_x:typing.Union[int, float]):

        return number_x \
            >> to_number_multiplied_by(number_other) \
            >> to_int_floored()

    return get_int_multiplied


def to_float_divided_by(
    number_other:typing.Union[int, float]):

    @unary()
    def inner(
        number_x:typing.Union[int, float]):

        return number_x / number_other

    return inner


def to_number_floor_divided_by(
    number_other:typing.Union[int, float]):

    @unary()
    def get_number_floor_divided_by(
        number_x:typing.Union[int, float]):

        return number_x // number_other

    return get_number_floor_divided_by


# TODO test further
def to_number_raised_to_power(
    number_power:typing.Union[int, float]):

    @unary()
    def inner(
        number_x:typing.Union[int, float]):

        return number_x ** number_power

    return inner


# TODO test
def to_number_minimum_with(
    number_other:typing.Union[int, float]):

    """
    Return the smaller number of the input number and "number_other"
    """

    @unary()
    def inner(
        number_x:typing.Union[int, float]):

        return min(
            number_x,
            number_other)

    return inner


# TODO test further
def to_number_maximum_with(
    number_other:typing.Union[int, float]):

    """
    Return the greater number of the input number and "number_other"
    """

    @unary()
    def inner(
        number_x:typing.Union[int, float]):

        return max(
            number_x,
            number_other)

    return inner


