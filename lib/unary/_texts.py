

from dataclasses import dataclass
import datetime
import typing

from lib.unary import _dicts
from lib.unary import _items
from lib.unary import _iters
from lib.unary import _tuples
from lib.unary.main import unary
from lib.unary.main import unary_pure




def to_text_append(
    text_other:str):

    @unary_pure()
    def get_concatenated(
        text:str):

        return text + text_other

    return get_concatenated


def to_text_prepend(
    text_other:str):

    @unary_pure()
    def get_text_prepended(
        text:str):

        return text_other + text

    return get_text_prepended


# TODO test further
def to_text_repeated(
    count_repetitions:int):

    @unary_pure()
    def inner(
        text:str):

        return text * count_repetitions

    return inner


def to_bool_is_empty():

    @unary_pure()
    def inner(
        text:str):

        return text == ""

    return inner


def to_bool_is_not_empty():

    @unary_pure()
    def inner(
        text:str):

        return text != ""

    return inner


# TODO test: test further
def to_bool_startswith(
    text_other:str):

    @unary_pure()
    def inner(
        text:str):

        return text \
            .startswith(text_other)

    return inner


def to_text_replace(
    text_old:str,
    text_new:str):

    @unary_pure()
    def replace(
        text:str):

        return text \
            .replace(
                text_old,
                text_new)

    return replace


def to_text_remove(
    text_to_remove:str):

    @unary_pure()
    def get_text_reduced(
        text:str):

        return text \
            .replace(
                text_to_remove,
                "")

    return get_text_reduced


def to_text_stripped(
    text_to_remove:typing.Optional[str] = None):

    @unary_pure()
    def strip(
        text:str):

        # TODO test: test when "text_to_remove" is None
        return text \
            .strip(text_to_remove)

    return strip


def to_text_stripped_right(
    text_to_remove:typing.Optional[str] = None):

    @unary_pure()
    def inner(
        text:str):

        # TODO test: test when "text_to_remove" is None
        return text \
            .rstrip(text_to_remove)

    return inner


def to_list_split_on(
    text_separator:typing.Optional[str] = None,
    number_splits_max:int = -1):

    @unary_pure()
    def get_list_split(
        text:str):

        return text \
            .split(
                text_separator,
                number_splits_max)

    return get_list_split


def to_tuple_partition_left(
    text_separator:str):

    @unary_pure()
    def inner(
        text:str):

        return text \
            .partition(text_separator)

    return inner


def to_tuple_partition_right(
    text_separator:str):

    @unary_pure()
    def inner(
        text:str):

        return text \
            .rpartition(text_separator)

    return inner


def to_text_left_of_leftmost_occurence_of(
    text_other:str):

    @unary_pure()
    def inner(
        text:str):

        return text \
            >> to_tuple_partition_left(text_other) \
            >> _tuples.to_item_first()

    return inner


def to_text_right_of_rightmost_occurence_of(
    text_other:str):

    @unary_pure()
    def inner(
        text:str):

        return text \
            >> to_tuple_partition_right(text_other) \
            >> _tuples.to_item_last()

    return inner


def to_int_count(
    text_to_find:str):

    @unary_pure()
    def get_count(
        text:str):

        return text \
            .count(text_to_find)

    return get_count


def to_text_lower():

    @unary_pure()
    def inner(
        text:str):

        return text \
            .lower()

    return inner


def to_text_upper():

    @unary_pure()
    def inner(
        text:str):

        return text \
            .upper()

    return inner


# TODO test further
def to_index_of(
    text_other:str,
    index_start:typing.Optional[typing.SupportsIndex] = None,
    index_end:typing.Optional[typing.SupportsIndex] = None):

    @unary_pure()
    def inner(
        text:str):

        return text \
            .index(
                text_other,
                index_start,
                index_end)

    return inner


def to_bool_does_contain(
    text_other:str):

    @unary_pure()
    def inner(
        text:str):

        return text_other in text

    return inner


def to_bool_does_not_contain(
    text_other:str):

    @unary_pure()
    def inner(
        text:str):

        return text_other not in text

    return inner


# TODO test further
def to_bool_is_in(
    text_other:str):

    @unary_pure()
    def inner(
        text:str):

        return text_other.__contains__(text)

    return inner


def to_text_quoted():

    @unary_pure()
    def get_text_quoted(
        text:str):

        return "\"" \
            + text \
            + "\""

    return get_text_quoted


def to_text_replace_multiple(
    dict_replacements:typing.Dict[str, str]):

    @dataclass(frozen=True)
    class Data_segment:
        text:str
        bool_is_replacement:bool

    @unary_pure()
    def get_segment_raw(
        text:str):

        return Data_segment(
            text=text,
            bool_is_replacement=False)

    @unary_pure()
    def get_segment_replacement(
        text:str):

        return Data_segment(
            text=text,
            bool_is_replacement=True)

    @unary_pure()
    def get_text(
        data_segment:Data_segment):

        return data_segment.text

    @unary_pure()
    def get_bool_is_replacement(
        data_segment:Data_segment):

        return data_segment.bool_is_replacement

    def to_iterable_segments_split(
        text_placeholder:str,
        text_replacement:str):

        @unary_pure()
        def get_iterable_segments_split(
            data_segment:Data_segment):

            if data_segment >> get_bool_is_replacement:
                return data_segment \
                    >> _items.to_iterable_singleton()
            else:
                return data_segment \
                    >> get_text \
                    >> to_list_split_on(text_placeholder) \
                    >> _iters.to_iterable_mapped(get_segment_raw) \
                    >> _iters.to_list_interleaved(text_replacement >> get_segment_replacement)

        return get_iterable_segments_split

    # TODO refactor: perhaps rename
    @unary_pure()
    def to_get_iterable_segments_replacement(
        pair_data:typing.Tuple[str, str]):

        return _iters.to_iterable_mapped(to_iterable_segments_split(
                text_placeholder=pair_data >> _tuples.to_item_first(),
                text_replacement=pair_data >> _tuples.to_item_last())) \
            | _iters.to_iterable_chained()

    def get_iterator_functions():
        return dict_replacements \
            >> _dicts.to_iterable_pairs() \
            >> _iters.to_iterable_mapped(to_get_iterable_segments_replacement) \
            >> _iters.to_iterator()

    @unary()
    def get_text_replace_multiple(
        text:str):

        return text \
            >> get_segment_raw \
            >> _items.to_iterable_singleton() \
            >> _items.to_item_apply_sequentially(get_iterator_functions()) \
            >> _iters.to_iterable_mapped(get_text) \
            >> _iters.to_text_joined("")

    return get_text_replace_multiple


# TODO test further
def to_datetime(
    text_format:str = "%Y-%m-%d"):

    @unary_pure()
    def inner(
        text:str):

        return datetime.datetime.strptime(
            text,
            text_format)

    return inner


# TODO implement: add argument: allow_default
# TODO implement: add argument: item_default
def to_int_parsed():

    @unary_pure()
    def inner(
        text:str):

        return int(text)

    return inner


def to_text_title():

    @unary_pure()
    def inner(
        text:str):

        return text \
            .title()

    return inner

