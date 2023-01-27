

from matplotlib import pyplot as plt
import pandas as pd

from lib.unary import _dicts
from lib.unary import _floats
from lib.unary import _ints
from lib.unary import _items
from lib.unary import _iters
from lib.unary import _lists
from lib.unary import _numbers
from lib.unary import _sets
from lib.unary import _sized
from lib.unary import _texts
from lib.unary import _tuples
from lib.unary.bs4 import _htmls
from lib.unary.main import unary




def plot_histogram(
    dataframe_statistics:pd.DataFrame,
    text_title:str,
    index_column:int,
    count_bins:int = 40):

    dataframe_statistics \
        .iloc[:, index_column] \
        .plot \
        .hist(
            bins=count_bins)

    plt.title(text_title)

    plt.show()


def print_statistics_categories(
    dataframe_data:pd.DataFrame):

    # TODO implement: print counts for each class

    def get_tuple_statistics_row(
        series_row:pd.Series):

        # TODO refactor
        count_professions_selected = (series_row \
            .iloc \
            [5:40] \
            != 0.0) \
            .sum()

        # TODO refactor
        # TODO refactor: duplicate code
        count_modes_employment_selected = (series_row \
            .iloc \
            [40:] \
            != 0.0) \
            .sum()

        return (
            count_professions_selected,
            count_modes_employment_selected)

    dataframe_statistics_categories = dataframe_data \
        .apply(
            func=get_tuple_statistics_row,
            axis=1,
            result_type="expand")

    plot_histogram(
            dataframe_statistics=dataframe_statistics_categories,
            text_title="Histogram: professions",
            index_column=0)

    plot_histogram(
            dataframe_statistics=dataframe_statistics_categories,
            text_title="Histogram: modes employment",
            index_column=1)

    float_average_count_professions_selected, \
    float_average_count_modes_employment_selected = dataframe_statistics_categories \
        .mean(
            axis=0) \
        >> _iters.to_tuple()

    text_average_professions_selected = float_average_count_professions_selected \
        >> _floats.to_text()

    text_average_modes_employment_selected = float_average_count_modes_employment_selected \
        >> _floats.to_text()

    def get_iterable_texts_statistics():

        # TODO implement: perhaps include standard deviations as well
        yield "\nAverages:\n  # professions:                       "
        yield text_average_professions_selected
        yield "\n  # modes employment:                  "
        yield text_average_modes_employment_selected

    text_statistics = get_iterable_texts_statistics() \
        >> _iters.to_text_joined("")

    print(text_statistics)


def print_statistics_keywords(
    dataframe_data:pd.DataFrame):

    @unary()
    def get_text_keyword_lower(
        text_raw:str):

        return text_raw \
            >> _texts.to_text_stripped() \
            >> _texts.to_text_lower()

    def get_tuple_statistics_row(
        series_row:pd.Series):

        text_offer_lower = series_row \
            .iloc \
            [3] \
            >> _texts.to_text_lower()

        set_texts_keywords_lower = series_row \
            .iloc \
            [4] \
            >> _texts.to_list_split_on(",") \
            >> _iters.to_iterable_mapped(get_text_keyword_lower) \
            >> _iters.to_set()

        count_words_document = text_offer_lower \
            >> _htmls.to_text_content() \
            >> _texts.to_list_split_on(" ") \
            >> _lists.to_int_length()

        count_keywords = set_texts_keywords_lower \
            >> _sized.to_int_length()

        # TODO error: when count_keywords is zero
        percentage_keywords_contained = set_texts_keywords_lower \
            >> _iters.to_iterable_filtered(_items.to_bool_is_in(text_offer_lower)) \
            >> _iters.to_list() \
            >> _lists.to_int_length() \
            >> _numbers.to_float_divided_by(count_keywords)

        return (
            count_words_document,
            count_keywords,
            percentage_keywords_contained)

    dataframe_statistics = dataframe_data \
        .apply(
            func=get_tuple_statistics_row,
            axis=1,
            result_type="expand")

    plot_histogram(
            dataframe_statistics=dataframe_statistics,
            text_title="Histogram: word count documents",
            index_column=0,
            count_bins=20)

    plot_histogram(
            dataframe_statistics=dataframe_statistics,
            text_title="Histogram: keyword count",
            index_column=1,
            count_bins=20)

    plot_histogram(
            dataframe_statistics=dataframe_statistics,
            text_title="Histogram: percentages keywords contained",
            index_column=2,
            count_bins=12)

    float_average_words_document, \
    float_average_keywords, \
    float_average_percentage_keywords_contained = dataframe_statistics \
        .mean(
            axis=0) \
        >> _iters.to_tuple()

    text_count_offers = dataframe_data \
        .shape \
        >> _tuples.to_item_first() \
        >> _ints.to_text()

    text_count_keywords_distinct = dataframe_data \
        .iloc \
        [:, 4] \
        >> _iters.to_iterable_mapped(_texts.to_list_split_on(",")) \
        >> _iters.to_iterable_chained() \
        >> _iters.to_iterable_mapped(_texts.to_text_stripped()) \
        >> _iters.to_set() \
        >> _sets.to_int_length() \
        >> _ints.to_text()

    text_average_count_words_document = float_average_words_document \
        >> _floats.to_text()

    text_average_count_keywords = float_average_keywords \
        >> _floats.to_text()

    text_average_percentage_keywords_contained = float_average_percentage_keywords_contained \
        >> _floats.to_text_as_percentage()

    def get_iterable_texts_statistics():

        # TODO implement: perhaps include standard deviations as well
        yield "Count offers:                          "
        yield text_count_offers
        yield "\nCount keywords distinct:               "
        yield text_count_keywords_distinct
        yield "\nAverages:\n  # words document:                    "
        yield text_average_count_words_document
        yield "\n  # keywords:                          "
        yield text_average_count_keywords
        yield "\n  % keywords contained (ignore case):  "
        yield text_average_percentage_keywords_contained
        yield "%"

    text_statistics = get_iterable_texts_statistics() \
        >> _iters.to_text_joined("")

    print(text_statistics)


def plot_distribution_keyword_position(
    dataframe_data:pd.DataFrame):

    COUNT_BINS_HISTOGRAM = 20

    def get_tuple_statistics_single(
        series_row:pd.Series):

        text_offer = series_row \
            .iloc \
            [3] \
            >> _htmls.to_text_content() \
            >> _texts.to_text_lower()

        int_length_text = text_offer \
            >> _sized.to_int_length()

        @unary()
        def get_float_position_relative(
            text_keyword:str):

            text_keyword_lower = text_keyword \
                >> _texts.to_text_lower()

            return text_offer \
                .find(text_keyword_lower) \
                >> _numbers.to_float_divided_by(int_length_text)

        dict_bins_incomplete = series_row \
            .iloc \
            [4] \
            >> _texts.to_list_split_on(", ") \
            >> _iters.to_iterable_mapped(get_float_position_relative) \
            >> _iters.to_dict_grouped(_numbers.to_int_multiplied_by(COUNT_BINS_HISTOGRAM)) \
            >> _dicts.to_dict_map_values(_sized.to_int_length())

        return range(
                -1,
                COUNT_BINS_HISTOGRAM) \
            >> _iters.to_iterable_mapped(
                function=_items.to_item_from_key_in(
                    dictionary=dict_bins_incomplete,
                    item_default=0)) \
            >> _iters.to_tuple()

    # TODO error: perhaps deviates from results in "print_statistics_keywords"
    series_counts = dataframe_data \
        .apply(
            func=get_tuple_statistics_single,
            axis=1,
            result_type="expand") \
        .sum(
            axis=0)

    count_keywords = series_counts \
        .sum()

    series_counts_relative = series_counts \
        / count_keywords

    list_indices = COUNT_BINS_HISTOGRAM \
        >> _ints.to_iterable_from_0() \
        >> _iters.to_iterable_mapped(_numbers.to_float_divided_by(COUNT_BINS_HISTOGRAM)) \
        >> _iters.to_iterable_prepend(-0.1) \
        >> _iters.to_list()

    plt.bar(
            x=list_indices,
            height=series_counts_relative,
            width=1 / COUNT_BINS_HISTOGRAM,
            align="edge")

    plt.xlabel("Relative position of keyword in text (-0.1 means not in text)")

    plt.ylabel("Frequency")

    plt.show()

