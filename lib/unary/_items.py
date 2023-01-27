

import typing

from lib.unary.main import Function_unary
from lib.unary.main import unary
from lib.unary.main import unary_pure




def to_item_unmodified():

    @unary_pure()
    def get_unmodified(
        item:typing.Any):

        return item

    return get_unmodified


# TODO change: perhaps remove
def to_item_apply_function_conventional(
    function_conventional:typing.Callable[[typing.Any], typing.Any]):

    @unary()
    def inner(
        item:typing.Any):

        return function_conventional(item)

    return inner


def to_bool_is_none():

    @unary_pure()
    def get_bool_is_none(
        item:typing.Any):

        return item is None

    return get_bool_is_none


def to_bool_is_not_none():

    @unary_pure()
    def get_bool_is_not_none(
        item:typing.Any):

        return item is not None

    return get_bool_is_not_none


def to_bool_is_equal_to(
    item_2:typing.Any):

    @unary_pure()
    def get_bool_is_equal_to(
        item:typing.Any):

        return item == item_2

    return get_bool_is_equal_to


def to_bool_is_not_equal_to(
    item_2:typing.Any):

    @unary_pure()
    def get_bool_is_not_equal_to(
        item:typing.Any):

        return item != item_2

    return get_bool_is_not_equal_to


def to_item_if(
    function_when:Function_unary,
    function_then:Function_unary = to_item_unmodified(),
    function_else:Function_unary = to_item_unmodified()):

    def get_bool_is_pure():

        return function_when.bool_is_pure \
            and function_then.bool_is_pure \
            and function_else.bool_is_pure

    @unary(get_bool_is_pure())
    def get_result(
        item:typing.Any):

        if item >> function_when:
            return item >> function_then
        else:
            return item >> function_else

    return get_result


def to_item_constant(
    item_constant:typing.Any):

    @unary_pure()
    def get_constant(
        item:typing.Any):

        return item_constant

    return get_constant


# TODO refactor: perhaps remove function
def to_bool_is_in(
    item_other:typing.Any):

    @unary()
    def get_bool_is_contained_in(
        item:typing.Any):

        return item in item_other

    return get_bool_is_contained_in


# TODO refactor: perhaps remove function
def to_bool_is_not_in(
    item_other:typing.Any):

    @unary()
    def get_bool_is_not_in(
        item:typing.Any):

        return item not in item_other

    return get_bool_is_not_in


# TODO refactor: perhaps rename
def to_item_from_key_in(
    dictionary:typing.Dict,
    do_allow_default:bool = True,
    item_default:typing.Any = None):

    # TODO refactor: duplicate code
    @unary()
    def get_item_from_key_in(
        item:typing.Any):

        if do_allow_default:
            return dictionary \
                .get(
                    item,
                    item_default)
        else:
            return dictionary \
                [item]

    return get_item_from_key_in


def to_text_default():

    @unary()
    def get_text(
        item:typing.Any):

        return str(item)

    return get_text


def to_iterable_repeated(
    count_repetitions:int):

    @unary()
    def get_iterable_repeated(
        item:typing.Any):

        for _ in range(count_repetitions):
            yield item

    return get_iterable_repeated


def to_iterable_singleton():

    return to_iterable_repeated(1)


def to_list_singleton():

    @unary()
    def get_list_singleton(
        item:typing.Any):

        return [item]

    return get_list_singleton


def to_pair(
    function_1:Function_unary = to_item_unmodified(),
    function_2:Function_unary = to_item_unmodified()):

    def get_bool_is_pure():

        return function_1.bool_is_pure \
            and function_2.bool_is_pure

    @unary(bool_is_pure=get_bool_is_pure())
    def get_pair(
        item:typing.Any):

        return (
            item >> function_1,
            item >> function_2)

    return get_pair


# TODO refactor: perhaps remove
def to_pair_with_none_second():

    @unary_pure()
    def get_pair_with_none_second(
        item:typing.Any):

        return (
            item,
            None)

    return get_pair_with_none_second


# TODO refactor: perhaps remove
def to_pair_with_none_first():

    @unary_pure()
    def get_pair_with_none_first(
        item:typing.Any):

        return (
            None,
            item)

    return get_pair_with_none_first


# TODO refactor: perhaps remove
def to_item_print():

    @unary()
    def get(
        item:typing.Any):

        # TODO implement: support keyword arguments
        print(item)

        return item

    return get


# TODO test: test further
def to_item_apply_sequentially(
    iterator_functions:typing.Iterator[Function_unary]):

    @unary()
    def get_result_sequential_application(
        item:typing.Any):

        try:
            # TODO refactor: perhaps refactor
            return item \
                >> next(iterator_functions) \
                >> get_result_sequential_application

        except StopIteration:
            return item

    return get_result_sequential_application

