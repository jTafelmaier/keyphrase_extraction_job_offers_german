

import typing

import pandas as pd

from lib.unary import _bools
from lib.unary import _dicts
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

from src.shared import _interface_os




PATH_RESOURCES_AI = _interface_os.get_path([
        "resources",
        "data_ml"])

NAME_FILE_CSV_DATA_ALL = "data_ml_all.csv"

NAME_FILE_CSV_DATA_TRAINING = "data_ml_training.csv"

NAME_FILE_CSV_DATA_VALIDATION = "data_ml_validation.csv"

NAME_FILE_CSV_DATA_TESTING = "data_ml_testing.csv"

PATH_CSV_DATA_ALL = _interface_os.get_path([
        PATH_RESOURCES_AI,
        NAME_FILE_CSV_DATA_ALL])

PATH_CSV_DATA_TRAINING = _interface_os.get_path([
        PATH_RESOURCES_AI,
        NAME_FILE_CSV_DATA_TRAINING])

PATH_CSV_DATA_VALIDATION = _interface_os.get_path([
        PATH_RESOURCES_AI,
        NAME_FILE_CSV_DATA_VALIDATION])

PATH_CSV_DATA_TESTING = _interface_os.get_path([
        PATH_RESOURCES_AI,
        NAME_FILE_CSV_DATA_TESTING])

NUMBER_CATEGORIES_PROFESSIONS = 35

NUMBER_CATEGORIES_MODES_EMPLOYMENT = 10


def get_list_names_subsidiaries_used_for_ai():

    # NOTE the names of the subsidiaries would be returned here (excluding platforms with non-german offers)
    raise NotImplementedError()


def get_path_dataframe_data_subsidiary(
    name_subsidiary_hrm:str):

    return name_subsidiary_hrm \
        >> _texts.to_text_replace(
            text_old=".",
            text_new="_") \
        >> _texts.to_text_prepend("resources/data_ml/data_ml_") \
        >> _texts.to_text_append(".csv")


def save_dataframes_data_subsidiaries():

    @unary()
    def get_iterable_dicts_offers(
        name_subsidiary_hrm:str):

        # NOTE offer data is retrieved from API here
        raise NotImplementedError()

    @unary()
    def save_dataframe_data(
        name_subsidiary_hrm:str):

        @unary()
        def get_tuple_data(
            dict_offer:typing.Dict):

            def get_set_indices_categories(
                key_type_categories:str):

                @unary()
                def get_id(
                    text_id:str):

                    return text_id \
                        >> _texts.to_tuple_partition_right("/") \
                        >> _tuples.to_item_last() \
                        >> _texts.to_int_parsed()

                return dict_offer \
                    >> _dicts.to_item_at(key_type_categories) \
                    >> _iters.to_iterable_mapped(get_id) \
                    >> _iters.to_set()

            set_indices_professions = get_set_indices_categories("professions")

            set_indices_employment_modes = get_set_indices_categories("modes employment")

            if set_indices_professions >> _sized.to_bool_is_empty():
                return None

            if set_indices_employment_modes >> _sized.to_bool_is_empty():
                return None

            set_indices_all = set_indices_employment_modes \
                >> _iters.to_iterable_mapped(_numbers.to_number_added_by(NUMBER_CATEGORIES_PROFESSIONS)) \
                >> _iters.to_set() \
                >> _sets.to_set_union(set_indices_professions)

            @unary()
            def get_float_is_in_set(
                index:bool):

                return index \
                    >> _items.to_item_if(
                        function_when=_items.to_bool_is_in(set_indices_all),
                        function_then=_items.to_item_constant(1.0),
                        function_else=_items.to_item_constant(0.0))

            dict_translation = dict_offer \
                >> _dicts.to_item_at("translations") \
                >> _lists.to_item_first()

            text_location = dict_translation \
                >> _dicts.to_item_at("location")

            list_keywords = dict_offer \
                >> _dicts.to_item_at("keywords") \
                >> _items.to_item_if(
                    function_when=_items.to_bool_is_none(),
                    function_then=_items.to_item_constant("")) \
                >> _texts.to_text_replace(
                    text_old=";",
                    text_new=",") \
                >> _texts.to_list_split_on(",") \
                >> _iters.to_iterable_mapped(_texts.to_text_stripped()) \
                >> _iters.to_iterable_filtered(_items.to_bool_is_not_equal_to(text_location)) \
                >> _iters.to_list()

            # NOTE removes duplicates while preserving order
            text_keywords = dict.fromkeys(list_keywords) \
                >> _dicts.to_iterable_keys() \
                >> _iters.to_text_joined(", ")

            if text_keywords >> _texts.to_bool_is_empty():
                return None

            @unary()
            def get_html_valid(
                key_attribute:str):

                html_raw = dict_translation \
                    >> _dicts.to_item_at(key_attribute)

                if html_raw >> _items.to_bool_is_none():
                    return None

                bool_is_empty = html_raw \
                    >> _htmls.to_text_content() \
                    >> _texts.to_text_stripped() \
                    >> _texts.to_bool_is_empty()

                if bool_is_empty:
                    return None

                return html_raw

            html_offer = [
                "html main",
                "html responsive"] \
                >> _iters.to_iterable_mapped(get_html_valid) \
                >> _iters.to_iterable_filter_none() \
                >> _iters.to_item_first()

            if html_offer >> _items.to_bool_is_none():
                return None

            bool_text_is_unusually_long = html_offer \
                >> _sized.to_int_length() \
                >> _numbers.to_bool_is_higher_than(10000)

            bool_might_be_pdf = html_offer \
                >> _texts.to_bool_does_contain("PDF") \
                >> _bools.to_bool_and(bool_text_is_unusually_long)

            if bool_might_be_pdf:
                print("Might be PDF: ID", dict_offer["id"])
                return None

            id_offer = dict_offer \
                >> _dicts.to_item_at("id offer")

            text_date_offer = dict_offer \
                >> _dicts.to_item_at("date posted")

            return NUMBER_CATEGORIES_PROFESSIONS \
                >> _numbers.to_number_added_by(NUMBER_CATEGORIES_MODES_EMPLOYMENT) \
                >> _ints.to_iterable_from_0() \
                >> _iters.to_iterable_mapped(get_float_is_in_set) \
                >> _iters.to_iterable_prepend(text_keywords) \
                >> _iters.to_iterable_prepend(html_offer) \
                >> _iters.to_iterable_prepend(text_date_offer) \
                >> _iters.to_iterable_prepend(id_offer) \
                >> _iters.to_iterable_prepend(name_subsidiary_hrm) \
                >> _iters.to_tuple()

        list_tuples_data = name_subsidiary_hrm \
            >> get_iterable_dicts_offers \
            >> _iters.to_iterable_filter_none() \
            >> _iters.to_iterable_mapped(get_tuple_data) \
            >> _iters.to_iterable_filter_none() \
            >> _iters.to_list()

        path_file = get_path_dataframe_data_subsidiary(name_subsidiary_hrm)

        pd.DataFrame(list_tuples_data) \
            .to_csv(
                path_or_buf=path_file,
                index=False)

    # TODO refactor: implement and use threaded for_each
    return get_list_names_subsidiaries_used_for_ai() \
        >> _iters.to_iterable_mapped(
            function=save_dataframe_data,
            do_force_parallelism=True) \
        >> _iters.to_list()


def save_dataframe_data_all_from_subsidiary_dataframes():

    @unary()
    def get_dataframe_data_subsidiary(
        name_subsidiary_hrm:str):

        path = get_path_dataframe_data_subsidiary(name_subsidiary_hrm)

        return pd.read_csv(
                filepath_or_buffer=path,
                keep_default_na=False)

    list_dataframes = get_list_names_subsidiaries_used_for_ai() \
        >> _iters.to_iterable_mapped(get_dataframe_data_subsidiary) \
        >> _iters.to_list()

    pd.concat(
            list_dataframes,
            axis=0) \
        .to_csv(
            path_or_buf=PATH_CSV_DATA_ALL,
            index=False)


def save_partitions_dataframe_data(
    bool_split_according_to_date:bool = False):

    def get_dataframe_data():

        dataframe_data = pd.read_csv(
                filepath_or_buffer=PATH_CSV_DATA_ALL,
                keep_default_na=False)

        if bool_split_according_to_date:
            # TODO implement: shuffle dataframe here as well
            return dataframe_data \
                .sort_values("2")
        else:
            return dataframe_data \
                .sample(
                    frac=1,
                    random_state=42)

    dataframe_data = get_dataframe_data()

    def get_int_cutoff(
        float_percentage:float):

        return int(dataframe_data \
            .__len__() \
            * float_percentage)

    def save_partition(
        path:str,
        index_start:typing.Optional[int],
        index_end:typing.Optional[int]):

        dataframe_data \
            .iloc \
            [index_start:index_end, :] \
            .to_csv(
                path_or_buf=path,
                index=False)

    int_cutoff_validation = get_int_cutoff(0.9)

    int_cutoff_testing = get_int_cutoff(0.95)

    save_partition(
            path=PATH_CSV_DATA_TRAINING,
            index_start=None,
            index_end=int_cutoff_validation)

    save_partition(
            path=PATH_CSV_DATA_VALIDATION,
            index_start=int_cutoff_validation,
            index_end=int_cutoff_testing)

    save_partition(
            path=PATH_CSV_DATA_TESTING,
            index_start=int_cutoff_testing,
            index_end=None)


def get_dataframe_data_partition(
    path_partition:str):

    return pd.read_csv(
            filepath_or_buffer=path_partition,
            keep_default_na=False)


def get_list_htmls(
    dataframe_data:pd.DataFrame):

    return dataframe_data \
        .iloc \
        [:, 3] \
        .to_list()


def get_list_texts_content_of_html(
    dataframe_data:pd.DataFrame):

    return get_list_htmls(dataframe_data) \
        >> _iters.to_iterable_mapped(_htmls.to_text_content()) \
        >> _iters.to_list()


def get_list_lists_texts_keywords(
    dataframe_data:pd.DataFrame):

    @unary()
    def get_list_texts_keywords_local(
        text_keywords:str):

        return text_keywords \
            >> _texts.to_list_split_on(",") \
            >> _iters.to_iterable_mapped(_texts.to_text_stripped()) \
            >> _iters.to_list()

    return dataframe_data \
        .iloc \
        [:, 4] \
        .to_list() \
        >> _iters.to_iterable_mapped(get_list_texts_keywords_local) \
        >> _iters.to_list()

