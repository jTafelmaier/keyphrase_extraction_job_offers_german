

import typing

from matplotlib import pyplot as plt
import evaluate as _hf_evaluate
import pandas as _pd

from lib.unary import _dicts
from lib.unary import _floats
from lib.unary import _items
from lib.unary import _iters
from lib.unary import _numbers
from lib.unary import _sets
from lib.unary import _sized
from lib.unary import _tuples
from lib.unary.bs4 import _htmls
from lib.unary.main import unary




def get_dataframe_evaluation(
    iterable_htmls:typing.Iterable[str],
    list_lists_texts_keywords_predicted:typing.List[typing.List[str]],
    list_lists_texts_keywords_target:typing.List[typing.List[str]]):

    metric_rouge = _hf_evaluate.load("rouge")

    def get_tuple_metrics(
        list_texts_keywords_target:typing.List[str],
        list_texts_keywords_predicted:typing.List[str]):

        set_texts_keywords_target = list_texts_keywords_target \
            >> _iters.to_set()

        set_texts_keywords_predicted = list_texts_keywords_predicted \
            >> _iters.to_set()

        count_keywords_predicted = set_texts_keywords_predicted \
            >> _sets.to_int_length()

        count_keywords_target = set_texts_keywords_target \
            >> _sets.to_int_length()

        count_keywords_matching = set_texts_keywords_predicted \
            >> _sets.to_set_intersection(set_texts_keywords_target) \
            >> _sets.to_int_length()

        def get_percentage_matching_relative_to_default_to_0(
            count_sequence_relative:int):

            # NOTE when count_sequence_relative == 0: score set to 0.0 to ensure consistency

            if count_sequence_relative >> _numbers.to_bool_is_equal_to(0):
                return 0.0
            else:
                return count_keywords_matching \
                    >> _numbers.to_float_divided_by(count_sequence_relative)

        float_recall = get_percentage_matching_relative_to_default_to_0(count_keywords_target)

        float_precision = get_percentage_matching_relative_to_default_to_0(count_keywords_predicted)

        def get_float_f1():

            # NOTE when float_sum_recall_precision == 0: score set to 0.0 to ensure consistency
            float_sum_recall_precision = float_recall \
                >> _numbers.to_number_added_by(float_precision)

            if float_sum_recall_precision >> _numbers.to_bool_is_equal_to(0.0):
                return 0.0
            else:
                return float_precision \
                    >> _numbers.to_number_multiplied_by(float_recall) \
                    >> _numbers.to_number_multiplied_by(2) \
                    >> _numbers.to_float_divided_by(float_sum_recall_precision)

        float_f1 = get_float_f1()

        text_keywords_predicted = list_texts_keywords_predicted \
            >> _iters.to_text_joined(", ")

        text_keywords_target = list_texts_keywords_target \
            >> _iters.to_text_joined(", ")

        # TODO refactor
        dict_rouge = metric_rouge \
            .compute(
                predictions=[text_keywords_predicted],
                references=[text_keywords_target])

        # NOTE for unigram tokenization performance without order
        float_rouge1 = dict_rouge \
            >> _dicts.to_item_at("rouge1")

        # NOTE for whole summary performance with order
        float_rouge_Lsum = dict_rouge \
            >> _dicts.to_item_at("rougeLsum")

        # TODO implement: perhaps evaluate BLEU or ROC-AUC as well
        return (
            float_f1,
            float_recall,
            float_precision,
            float_rouge1,
            float_rouge_Lsum)

    @unary()
    def get_tuple_data_evaluated(
        tuple_data:typing.Tuple[str, typing.List[str], typing.List[str]]):

        html, \
        list_texts_keywords_target_all, \
        list_texts_keywords_predicted_all = tuple_data

        text_content_html = html \
            >> _htmls.to_text_content()

        @unary()
        def get_list_pairs_keywords_about_containment(
            list_texts_keywords:typing.List[str]):

            return list_texts_keywords \
                >> _iters.to_iterable_mapped(
                    function=_items.to_pair(
                        function_1=_items.to_bool_is_in(text_content_html))) \
                >> _iters.to_list()

        list_pairs_keywords_target = list_texts_keywords_target_all \
            >> get_list_pairs_keywords_about_containment

        list_pairs_keywords_predicted = list_texts_keywords_predicted_all \
            >> get_list_pairs_keywords_about_containment

        def get_list_texts_keywords_filtered(
            list_pairs_keywords:typing.List[typing.Tuple[bool, str]],
            bool_should_be_contained:bool):

            @unary()
            def get_bool_matches_condition(
                pair_keyword:typing.Tuple[bool, str]):

                return pair_keyword \
                    >> _tuples.to_item_first() \
                    >> _items.to_bool_is_equal_to(bool_should_be_contained)

            return list_pairs_keywords \
                >> _iters.to_iterable_filtered(get_bool_matches_condition) \
                >> _iters.to_iterable_mapped(_tuples.to_item_last()) \
                >> _iters.to_list()

        list_texts_keywords_target_contained = get_list_texts_keywords_filtered(
                list_pairs_keywords=list_pairs_keywords_target,
                bool_should_be_contained=True)

        list_texts_keywords_target_not_contained = get_list_texts_keywords_filtered(
                list_pairs_keywords=list_pairs_keywords_target,
                bool_should_be_contained=False)

        list_texts_keywords_predicted_contained = get_list_texts_keywords_filtered(
                list_pairs_keywords=list_pairs_keywords_predicted,
                bool_should_be_contained=True)

        list_texts_keywords_predicted_not_contained = get_list_texts_keywords_filtered(
                list_pairs_keywords=list_pairs_keywords_predicted,
                bool_should_be_contained=False)

        tuple_metrics_contained = get_tuple_metrics(
                list_texts_keywords_target=list_texts_keywords_target_contained,
                list_texts_keywords_predicted=list_texts_keywords_predicted_contained)

        tuple_metrics_not_contained = get_tuple_metrics(
                list_texts_keywords_target=list_texts_keywords_target_not_contained,
                list_texts_keywords_predicted=list_texts_keywords_predicted_not_contained)

        tuple_metrics_all = get_tuple_metrics(
                list_texts_keywords_target=list_texts_keywords_target_all,
                list_texts_keywords_predicted=list_texts_keywords_predicted_all)

        return tuple_data \
            >> _iters.to_iterable_extended(tuple_metrics_all) \
            >> _iters.to_iterable_extended(tuple_metrics_contained) \
            >> _iters.to_iterable_extended(tuple_metrics_not_contained) \
            >> _iters.to_tuple()

    list_tuples_evaluation = iterable_htmls \
        >> _iters.to_iterable_zipped_with_multiple([
            list_lists_texts_keywords_target,
            list_lists_texts_keywords_predicted]) \
        >> _iters.to_iterable_mapped(get_tuple_data_evaluated) \
        >> _iters.to_list()

    return _pd.DataFrame(list_tuples_evaluation)


def get_dict_metrics_average(
    dataframe_evaluation:_pd.DataFrame):

    list_texts_names_metrics = [
        "All:           f1        ",
        "All:           recall    ",
        "All:           precision ",
        "All:           rouge1    ",
        "All:           rougeLsum ",
        "Contained:     f1        ",
        "Contained:     recall    ",
        "Contained:     precision ",
        "Contained:     rouge1    ",
        "Contained:     rougeLsum ",
        "Not contained: f1        ",
        "Not contained: recall    ",
        "Not contained: precision ",
        "Not contained: rouge1    ",
        "Not contained: rougeLsum "]

    # TODO implement: perhaps compute standard deviations
    list_texts_metrics_average = dataframe_evaluation \
        .iloc \
        [:, 3:] \
        .mean(
            axis=0) \
        .to_list() \
        >> _iters.to_iterable_mapped(
            function=_floats.to_text(
                int_precision=3)) \
        >> _iters.to_list()

    return list_texts_names_metrics \
        >> _iters.to_iterable_zipped(list_texts_metrics_average) \
        >> _iters.to_dict()


def plot_correlations_of_f1(
    dataframe_evaluation:_pd.DataFrame,
    bool_plot_distributions:bool = False):

    list_scores_f1 = dataframe_evaluation \
        .iloc \
        [:, 3] \
        .to_list()

    def plot_histogram_distribution(
        list_numbers:typing.List[int],
        text_description:str,
        count_bins:int = 20):

        # TODO implement: display relative y values
        plt.hist(
                x=list_numbers,
                bins=count_bins)

        plt.xlabel(text_description)

        plt.ylabel("Frequency")

        plt.show()

    def plot_correlation_to_f1(
        list_ints_values:typing.List[int],
        text_description_values:str):

        if bool_plot_distributions:
            plot_histogram_distribution(
                    list_numbers=list_ints_values,
                    text_description=text_description_values)

        # NOTE error: normal distribution of values assumed
        dict_pearson_r = _hf_evaluate.load("pearsonr") \
            .compute(
                predictions=list_ints_values,
                references=list_scores_f1,
                return_pvalue=True)

        text_correlation = dict_pearson_r \
            >> _dicts.to_item_at("pearsonr") \
            >> _floats.to_text(3)

        text_pvalue = dict_pearson_r \
            >> _dicts.to_item_at("p-value") \
            >> _floats.to_text(3)

        text_title = "Pearson correlation: " \
            + text_correlation \
            + ", p-value: " \
            + text_pvalue

        plt.scatter(
                x=list_ints_values,
                y=list_scores_f1)

        plt.xlabel(text_description_values)

        plt.ylabel("F1 score")

        plt.title(text_title)

        plt.show()

    if bool_plot_distributions:
        plot_histogram_distribution(
                list_numbers=list_scores_f1,
                text_description="F1 score")

    @unary()
    def get_count_characters_in_text(
        html:str):

        return html \
            >> _htmls.to_text_content() \
            >> _sized.to_int_length()

    # TODO implement: perhaps correlate number of tokens instead of character counts
    list_counts_characters = dataframe_evaluation \
        .iloc \
        [:, 0] \
        >> _iters.to_iterable_mapped(get_count_characters_in_text) \
        >> _iters.to_list()

    list_counts_keywords_target = dataframe_evaluation \
        .iloc \
        [:, 1] \
        >> _iters.to_iterable_mapped(_sized.to_int_length()) \
        >> _iters.to_list()

    plot_correlation_to_f1(
            list_ints_values=list_counts_characters,
            text_description_values="Input text length (number of characters)")

    plot_correlation_to_f1(
            list_ints_values=list_counts_keywords_target,
            text_description_values="Target sequence length")

