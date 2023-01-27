

import functools
import operator
import statistics
import sys
import traceback
import typing
from concurrent import futures

from lib.unary import _dicts
from lib.unary import _items
from lib.unary import _lists
from lib.unary import _sequences
from lib.unary.main import Function_unary
from lib.unary.main import unary




def to_iterable_mapped(
    function:Function_unary,
    do_force_parallelism:bool = False):

    """
    :return: an iterable containing function(x) for each x in the input iterable (similar to map(function, iterable))
    :argument "do_force_parallelism": if true, use threading to force parallelism (currently only advantageous on io operations)
    :note if "do_force_parallelism" is True: input iterable is eagerly collected (not lazy)!
    """

    @unary()
    def get_map_sequential(
        iterable:typing.Iterable):

        return map(
            function,
            iterable)

    def get_result_try(
        item:typing.Any):

        try:
            # TODO refactor: use "item >> function" when sufficiently prepared (test for some time with "assert isinstance(function, Function_unary)")
            return function(item)

        except Exception:
            sys.stdout.write("futures.ThreadPoolExecutor(): Exception in Thread.:\n" \
                + traceback.format_exc() \
                + "\n")
            # TODO implement: perhaps try to raise Exception outside of thread instead
            return None

    @unary()
    def get_map_parallel(
        iterable:typing.Iterable):

        executor = futures.ThreadPoolExecutor()

        # TODO refactor: perhaps use yield from
        with executor:
            return executor \
                .map(
                    get_result_try,
                    iterable)

    # NOTE could add "or function.bool_is_pure", however since Python is single-threaded, this is not done
    if do_force_parallelism:
        return get_map_parallel
    else:
        return get_map_sequential


def to_iterable_filtered(
    function:Function_unary,
    do_force_parallelism:bool = False):

    # TODO implement: remove later
    assert isinstance(function, Function_unary)

    @unary()
    def get_iterable_filter_relevant(
        iterable:typing.Iterable):

        return filter(
            _sequences.to_item_last(),
            iterable)

    @unary()
    def inner(
        iterable:typing.Iterable):

        return iterable \
            >> to_iterable_mapped(
                function=_items.to_pair(
                    function_2=function),
                do_force_parallelism=do_force_parallelism) \
            >> get_iterable_filter_relevant \
            >> to_iterable_mapped(_sequences.to_item_first())

    return inner


# TODO test further
# TODO refactor: perhaps rename
def to_item_reduce(
    function_binary:typing.Callable[[typing.Any, typing.Any], typing.Any],
    item_initial:typing.Any):

    @unary()
    def get_reduced(
        iterable:typing.Iterable):

        return functools.reduce(
            function_binary,
            iterable,
            item_initial)

    return get_reduced


# TODO test further
def to_number_sum():

    return to_item_reduce(
        function_binary=operator.add,
        item_initial=0)


def to_iterable_chained():

    @unary()
    def get_iterable_chained(
        iterable:typing.Iterable[typing.Iterable]):

        for iterable_inner in iterable:
            yield from iterable_inner

    return get_iterable_chained


def to_iterable_until(
    function_condition:Function_unary,
    bool_include_last_item:bool = False):

    @unary()
    def get_iterable_until(
        iterable:typing.Iterable):

        for item in iterable:
            if item >> function_condition:
                if bool_include_last_item:
                    yield item
                break
            else:
                yield item

    return get_iterable_until


def to_iterable_zipped(
    iterable_2:typing.Iterable):

    @unary()
    def inner(
        iterable:typing.Iterable):

        return zip(
            iterable,
            iterable_2)

    return inner


# TODO test: test further
def to_iterable_zipped_with_multiple(
    iterable_iterables:typing.Iterable[typing.Iterable]):

    @unary()
    def inner(
        iterable:typing.Iterable):

        return zip(
            iterable,
            *iterable_iterables)

    return inner


# TODO refactor: rename: "to_iterable_extended_with"
def to_iterable_extended(
    iterable_2:typing.Iterable):

    @unary()
    def get_iterable_extended(
        iterable:typing.Iterable):

        yield from iterable
        yield from iterable_2

    return get_iterable_extended


def to_dict_grouped(
    function_get_key:Function_unary,
    function_get_value:Function_unary = _items.to_item_unmodified()):

    @unary()
    def get_dict_grouped(
        iterable:typing.Iterable):

        dict_grouped:typing.Dict = {}

        for key, item in iterable \
            >> to_iterable_mapped(
                function=_items.to_pair(
                    function_1=function_get_key,
                    function_2=function_get_value)):

            list_previous = dict_grouped \
                >> _dicts.to_item_get(key)

            if list_previous is None:
                dict_grouped[key] = [item]
            else:
                list_previous \
                    .append(item)

        return dict_grouped

    return get_dict_grouped


def to_iterable_prepend(
    item:typing.Any):

    @unary()
    def get_iterable_with_prepend(
        iterable:typing.Iterable):

        yield item
        yield from iterable

    return get_iterable_with_prepend


def to_iterable_append(
    item:typing.Any):

    @unary()
    def get_iterable_with_append(
        iterable:typing.Iterable):

        yield from iterable
        yield item

    return get_iterable_with_append


def to_iterator_drop_items( # TODO test further
    count_items_to_drop:int,
    do_ignore_stopiteration:bool = False):

    @unary()
    def get_iterable_reduced(
        iterable:typing.Iterable):

        iterator = iter(iterable)

        try:
            for _ in range(count_items_to_drop):
                next(iterator) # TODO try to refactor
        except StopIteration:
            if not do_ignore_stopiteration:
                raise Exception("to_iterator_drop_items: more items to drop than available")

        return iterator

    return get_iterable_reduced


def to_iterator_drop_first(
    do_ignore_stopiteration:bool = False):

    return to_iterator_drop_items(
        count_items_to_drop=1,
        do_ignore_stopiteration=do_ignore_stopiteration)


def to_iterable_limit_count(
    count_items_max:int):

    @unary()
    def inner(
        iterable:typing.Iterable):

        for i, item in enumerate(iterable):
            if i >= count_items_max:
                return
            else:
                yield item

    return inner


# TODO test further
def to_iterable_alternating(
    iterable_2:typing.Iterable):

    return to_iterable_zipped(iterable_2) \
        | to_iterable_chained()


# TODO test further
def to_list_interleaved(
    item:typing.Any):

    @unary()
    def get_iterable_interleaved(
        iterable:typing.Iterable):

        list_main = iterable \
            >> to_list()

        iterable_item_repeated = item \
            >> _items.to_iterable_repeated(list_main >> _lists.to_int_length())

        return list_main \
            >> to_iterable_alternating(iterable_item_repeated) \
            >> to_list() \
            >> _lists.to_list_slice(index_end=-1)

    return get_iterable_interleaved


def to_list_sorted(
    function_get_key:Function_unary = None,
    do_sort_descending:bool = False):

    @unary()
    def get_sorted(
        iterable:typing.Iterable):

        return sorted(
            iterable,
            key=function_get_key,
            reverse=do_sort_descending)

    return get_sorted


# TODO refactor: rename
def to_iterable_filter_none(
    function_get_key:Function_unary = _items.to_item_unmodified()):

    @unary()
    def get_bool_is_not_none_key(
        item:typing.Any):

        return item \
            >> function_get_key \
            >> _items.to_bool_is_not_none()

    return to_iterable_filtered(get_bool_is_not_none_key)


def to_iterator():

    @unary()
    def get_iterator(
        iterable:typing.Iterable):

        return iter(iterable)

    return get_iterator


def to_list():

    """
    :return: A list containing all items of the iterable
    :note Synchronizes previous lazy operations such as "to_iterable_mapped" since all items are collected
    """

    @unary()
    def get_list(
        iterable:typing.Iterable):

        return list(iterable)

    return get_list


def to_tuple():

    @unary()
    def get_tuple(
        iterable:typing.Iterable):

        return tuple(iterable)

    return get_tuple


def to_set():

    @unary()
    def get_set(
        iterable:typing.Iterable):

        return set(iterable)

    return get_set


def to_frozenset():

    @unary()
    def get_frozenset(
        iterable:typing.Iterable):

        return frozenset(iterable)

    return get_frozenset


def to_dict():

    @unary()
    def get_dict(
        iterable:typing.Iterable[typing.Tuple[typing.Any, typing.Any]]):

        return dict(iterable)

    return get_dict


# TODO test further
def to_iterable_reversed():

    @unary()
    def get_reversed(
        iterable:typing.Sequence):

        return reversed(list(iterable))

    return get_reversed


def to_iterable_for_each_lazy(
    function:Function_unary):

    @unary()
    def get_iterable_items(
        iterable:typing.Iterable):

        for item in iterable:
            # TODO use "item >> function" when sufficiently prepared (test for some time with "assert isinstance(function, Function_unary)")
            function(item)
            yield item

    return get_iterable_items


def to_iterable_for_each_eager(
    function:Function_unary):

    @unary()
    def get_list(
        iterable:typing.Iterable):

        return iterable \
            >> to_iterable_for_each_lazy(function) \
            >> to_list()

    return get_list


def to_iterable_enumerated():

    @unary()
    def get_iterable_enumerated(
        iterable:typing.Iterable):

        return enumerate(iterable)

    return get_iterable_enumerated


# TODO test further
# TODO refactor: rename: "to_iterable_swap_first_2_dimensions" or "to_iterable_transpose_first_2_dimensions"
def to_iterable_swap_dimensions():

    @unary()
    def get_iterable_swapped_dimensions(
        iterable:typing.Iterable):

        return zip(*iterable)

    return get_iterable_swapped_dimensions


def to_item_first(
    do_allow_default:bool = True,
    item_default:typing.Any = None):

    @unary()
    def get_first_default(
        iterable:typing.Iterable):

        return next(
            iter(iterable),
            item_default)

    @unary()
    def get_first_strict(
        iterable:typing.Iterable):

        return next(iter(iterable))

    if do_allow_default:
        return get_first_default
    else:
        return get_first_strict


def to_bool_is_not_empty():

    # TODO test: further testing
    # NOTE modifies the iterable

    @unary()
    def inner(
        iterable:typing.Iterable):

        try:
            next(iter(iterable))
            return True
        except StopIteration:
            return False

    return inner


# TODO test further
def to_float_average():

    @unary()
    def get_float_average(
        iterable:typing.Iterable):

        return statistics \
            .mean(iterable)

    return get_float_average


def to_bool_any():

    @unary()
    def get_any(
        iterable:typing.Iterable):

        return any(iterable)

    return get_any


# TODO test further
def to_bool_all():

    @unary()
    def get_all(
        iterable:typing.Iterable):

        return all(iterable)

    return get_all


# TODO refactor: perhaps rename: "to_text_joined_by"
def to_text_joined(
    text_separator:str):

    @unary()
    def get_text_joined(
        iterable:typing.Iterable[str]):

        return text_separator.join(iterable)

    return get_text_joined

