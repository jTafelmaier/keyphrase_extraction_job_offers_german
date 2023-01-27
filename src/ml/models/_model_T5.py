

import typing

import datasets
import torch
import transformers
from matplotlib import pyplot as plt
from transformers import trainer_utils

from lib.unary import _dicts
from lib.unary import _items
from lib.unary import _iters
from lib.unary import _jsons
from lib.unary import _numbers
from lib.unary import _texts
from lib.unary import _tuples
from lib.unary.bs4 import _htmls
from lib.unary.main import unary

from src.ml import _ml_retrieval




# TODO implement: perhaps vary
LENGTH_INPUT = 512

LENGTH_OUTPUT = 128

ID_TOKEN_IGNORE_IN_LOSS_PYTORCH = -100

NAME_CHECKPOINT = "resources/data_ml/data_T5"

# TODO implement: perhaps create unique prefix
TEXT_PREFIX = "summarize: "


# NOTE parts taken from "https://huggingface.co/docs/transformers/v4.25.1/en/tasks/summarization"
def train_new_model(
    count_epochs:int = 20):

    NAME_CHECKPOINT_PRETRAINED = "t5-small"

    SIZE_BATCHES_GPU = 10

    tokenizer = transformers.T5Tokenizer \
        .from_pretrained(
            pretrained_model_name_or_path=NAME_CHECKPOINT_PRETRAINED,
            model_max_length=LENGTH_INPUT)

    # TODO implement: perhaps use "AutoModelForSeq2SeqLM"
    model = transformers.T5ForConditionalGeneration \
        .from_pretrained(NAME_CHECKPOINT_PRETRAINED)

    # TODO investigate: research and try different strategies
    strategy_padding = transformers.utils.PaddingStrategy \
        .MAX_LENGTH

    name_column_input = "3"

    name_column_target = "4"

    # TODO implement: add test data
    # TODO test: add argument: "cache_dir"
    dataset_dict_raw = datasets.load_dataset(
        path=_ml_retrieval.PATH_RESOURCES_AI,
        data_files={
            "train": _ml_retrieval.NAME_FILE_CSV_DATA_TRAINING,
            "validation": _ml_retrieval.NAME_FILE_CSV_DATA_VALIDATION})

    def get_batch_encoding_processed(
        batch_data:datasets.arrow_dataset.Batch):

        # TODO refactor: move
        def to_batch_encoding_tokenized(
            tokenizer:transformers.T5Tokenizer,
            strategy_padding:transformers.utils.PaddingStrategy,
            int_length_max:typing.Optional[int],
            do_truncate:bool):

            @unary()
            def inner(
                list_texts:typing.List[str]):

                return tokenizer \
                    .__call__(
                        text=list_texts,
                        padding=strategy_padding,
                        max_length=int_length_max,
                        truncation=do_truncate)

            return inner

        @unary()
        def get_list_replace_id_padding(
            list_ids_tokens:typing.List[int]):

            return list_ids_tokens \
                >> _iters.to_iterable_mapped(
                    function=_items.to_item_if(
                        function_when=_items.to_bool_is_equal_to(tokenizer.pad_token_id),
                        function_then=_items.to_item_constant(ID_TOKEN_IGNORE_IN_LOSS_PYTORCH))) \
                >> _iters.to_list()

        dict_data = batch_data \
            .data

        list_texts_target = dict_data \
            >> _dicts.to_item_at(name_column_target)

        list_texts_inputs_raw, \
        list_texts_targets = dict_data \
            >> _dicts.to_item_at(name_column_input) \
            >> _iters.to_iterable_mapped(_htmls.to_text_content()) \
            >> _iters.to_iterable_zipped(list_texts_target) \
            >> _iters.to_iterable_swap_dimensions() \
            >> _iters.to_iterable_mapped(_iters.to_list())

        list_lists_ids_tokens_labels = list_texts_targets \
            >> to_batch_encoding_tokenized(
                tokenizer=tokenizer,
                strategy_padding=strategy_padding,
                int_length_max=LENGTH_OUTPUT,
                do_truncate=True) \
            >> _dicts.to_item_at("input_ids") \
            >> _iters.to_iterable_mapped(get_list_replace_id_padding) \
            >> _iters.to_list()

        return list_texts_inputs_raw \
            >> _iters.to_iterable_mapped(_texts.to_text_prepend(TEXT_PREFIX)) \
            >> _iters.to_list() \
            >> to_batch_encoding_tokenized(
                tokenizer=tokenizer,
                strategy_padding=strategy_padding,
                int_length_max=None,
                do_truncate=True) \
            >> _dicts.to_dict_inplace_set(
                item_key="labels",
                item_value=list_lists_ids_tokens_labels)

    def get_dataset_tokenized(
        text_key:str):

        dataset_raw = dataset_dict_raw \
            [text_key]

        text_description = "Running tokenizer on " \
            + text_key \
            + " dataset"

        return dataset_raw \
            .map(
                function=get_batch_encoding_processed,
                remove_columns=dataset_raw.column_names,
                batched=True,
                load_from_cache_file=True,
                desc=text_description)

    dataset_train = get_dataset_tokenized("train")

    dataset_validation = get_dataset_tokenized("validation")

    # def postprocess_text(
    #     preds,
    #     labels):

    #     preds = [pred.strip() for pred in preds]
    #     labels = [label.strip() for label in labels]

    #     # NOTE rougeLSum expects newline after each sentence
    #     preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    #     labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    #     return (
    #         preds,
    #         labels)

    # # TODO refactor
    # def compute_metrics(
    #     eval_preds:trainer_utils.EvalPrediction):

    #     metric = datasets.load_metric("rouge")

    #     preds, \
    #     labels = eval_preds

    #     if isinstance(preds, tuple):
    #         preds = preds[0]

    #     # TODO ERROR: shape of "preds" is incorrect
    #     decoded_preds = tokenizer.batch_decode(
    #         sequences=preds,
    #         skip_special_tokens=True)

    #     # NOTE Replace -100 in the labels as we can't decode them.
    #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    #     decoded_labels = tokenizer.batch_decode(
    #         sequences=labels,
    #         skip_special_tokens=True)

    #     decoded_preds, \
    #     decoded_labels = postprocess_text(
    #         preds=decoded_preds,
    #         labels=decoded_labels)

    #     dict_result = metric.compute(
    #         predictions=decoded_preds,
    #         references=decoded_labels,
    #         use_stemmer=True)

    #     # NOTE Extract a few results from ROUGE
    #     dict_result = {key: value.mid.fmeasure * 100 for key, value in dict_result.items()}

    #     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

    #     dict_result["gen_len"] = np.mean(prediction_lens)

    #     return {k: round(v, 4) for k, v in dict_result.items()}

    # TODO investigate: measure impact on performance
    # TODO investigate: purpose
    data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=ID_TOKEN_IGNORE_IN_LOSS_PYTORCH,
            pad_to_multiple_of=None)

    # TODO refactor
    # TODO refactor: perhaps set "compute_metrics" to None
    # TODO test: test argument "eval_accumulation_steps"
    # TODO test: test argument "eval_delay"
    trainer = transformers.Seq2SeqTrainer(
            model=model,
            args=transformers.Seq2SeqTrainingArguments(
                output_dir="resources/data_ml/data_training",
                overwrite_output_dir=True,
                per_device_train_batch_size=SIZE_BATCHES_GPU,
                per_device_eval_batch_size=SIZE_BATCHES_GPU,
                num_train_epochs=count_epochs,
                learning_rate=1e-4,
                optim=transformers.training_args.OptimizerNames.ADAFACTOR,
                evaluation_strategy=trainer_utils.IntervalStrategy.STEPS,
                eval_steps=10000,
                # eval_accumulation_steps=2,
                eval_delay=50000,
                save_strategy=trainer_utils.IntervalStrategy.STEPS,
                save_steps=10000,
                save_total_limit=3,
                bf16=True),
            train_dataset=dataset_train,
            eval_dataset=dataset_validation,
            tokenizer=tokenizer,
            data_collator=data_collator,
            # compute_metrics=compute_metrics
            )

    # TODO implement: perhaps use "trainer_utils.get_last_checkpoint("resources/data_ml/data_training")"
    train_result = trainer \
        .train(
            resume_from_checkpoint=None)

    metrics = train_result \
        .metrics

    metrics["train_samples"] = len(dataset_train)

    trainer \
        .log_metrics(
            split="train",
            metrics=metrics)

    metrics_evaluation = trainer \
        .evaluate(
            eval_dataset=dataset_validation)

    trainer \
        .log_metrics(
            split="validation",
            metrics=metrics_evaluation)


# TODO refactor: integrate
def get_list_lists_texts_keywords_batch(
    list_htmls_batch:typing.List[str],
    model_t5:transformers.T5ForConditionalGeneration,
    tokenizer_t5:transformers.T5Tokenizer,
    device:torch.device):

    """
    NOTE batch size cannot be arbitrarily large.
    """

    @unary()
    def get_text_input(
        html:str):

        return html \
            >> _htmls.to_text_content() \
            >> _texts.to_text_prepend(TEXT_PREFIX)

    list_texts_input = list_htmls_batch \
        >> _iters.to_iterable_mapped(get_text_input) \
        >> _iters.to_list()

    tensor_ids_input = tokenizer_t5 \
        .batch_encode_plus(
            batch_text_or_text_pairs=list_texts_input,
            return_tensors="pt",
            padding=True,
            truncation=True) \
        ["input_ids"] \
        .to(device)

    greedy_output = model_t5 \
        .generate(
            inputs=tensor_ids_input,
            max_length=200)

    @unary()
    def get_list_texts_keywords_local(
        text_keywords_raw:str):

        # NOTE using "to_dict_grouped" to remove duplicates while retaining order
        # TODO error: in output: words with "Ö" may be unknown to tokenizer (map to "unk")
        # TODO error: investigate (previous) higher occurence of "Tiroler Unterland", "Tiroler Oberland" or "Krankenschw". Check for frequency of these keywords in database.
        return text_keywords_raw \
            >> _texts.to_list_split_on(",") \
            >> _iters.to_iterable_mapped(_texts.to_text_stripped()) \
            >> _iters.to_iterable_filtered(_texts.to_bool_is_not_empty()) \
            >> _iters.to_dict_grouped(_items.to_item_unmodified()) \
            >> _dicts.to_iterable_keys() \
            >> _iters.to_list()

    return tokenizer_t5 \
        .batch_decode(
            sequences=greedy_output,
            skip_special_tokens=True) \
        >> _iters.to_iterable_mapped(get_list_texts_keywords_local) \
        >> _iters.to_list()


def get_list_lists_texts_keywords(
    iterable_htmls:typing.Iterable[str],
    model_t5:transformers.T5ForConditionalGeneration,
    tokenizer_t5:transformers.T5Tokenizer,
    device:torch.device):

    INT_SIZE_BATCHES = 10

    @unary()
    def get_list_lists_texts_keywords_local(
        list_htmls_batch:typing.List[str]):

        return get_list_lists_texts_keywords_batch(
                list_htmls_batch=list_htmls_batch,
                model_t5=model_t5,
                tokenizer_t5=tokenizer_t5,
                device=device)

    return iterable_htmls \
        >> _iters.to_iterable_enumerated() \
        >> _iters.to_iterable_mapped(
            function=_tuples.to_pair_mapped(
                function_first=_numbers.to_number_floor_divided_by(INT_SIZE_BATCHES))) \
        >> _iters.to_dict_grouped(
            function_get_key=_tuples.to_item_first(),
            function_get_value=_tuples.to_item_last()) \
        >> _dicts.to_iterable_values() \
        >> _iters.to_iterable_mapped(get_list_lists_texts_keywords_local) \
        >> _iters.to_iterable_chained() \
        >> _iters.to_list()


def get_list_lists_texts_keywords_load_model(
    iterable_htmls:typing.Iterable[str]):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO implement: perhaps try to use cuda here
    tokenizer_t5 = transformers.T5Tokenizer \
        .from_pretrained(
            pretrained_model_name_or_path=NAME_CHECKPOINT,
            model_max_length=LENGTH_INPUT)

    model_t5 = transformers.T5ForConditionalGeneration \
        .from_pretrained(NAME_CHECKPOINT) \
        .to(device)

    return get_list_lists_texts_keywords(
            iterable_htmls=iterable_htmls,
            model_t5=model_t5,
            tokenizer_t5=tokenizer_t5,
            device=device)


def plot_progress_loss():

    list_dicts_log = "resources\\data_ml\\data_T5\\trainer_state.json" \
        >> _jsons.to_dict_json_from_path() \
        >> _dicts.to_item_at("log_history")

    list_dicts_training = list_dicts_log \
        >> _iters.to_iterable_filter_none(_dicts.to_item_get("loss")) \
        >> _iters.to_list()

    list_dicts_evaluation = list_dicts_log \
        >> _iters.to_iterable_filter_none(_dicts.to_item_get("eval_loss")) \
        >> _iters.to_list()

    list_floats_loss_training = list_dicts_training \
        >> _iters.to_iterable_mapped(_dicts.to_item_at("loss")) \
        >> _iters.to_list()

    list_floats_epochs_training = list_dicts_training \
        >> _iters.to_iterable_mapped(_dicts.to_item_at("epoch")) \
        >> _iters.to_list()

    list_floats_loss_evaluation = list_dicts_evaluation \
        >> _iters.to_iterable_mapped(_dicts.to_item_at("eval_loss")) \
        >> _iters.to_list()

    list_floats_epochs_evaluation = list_dicts_evaluation \
        >> _iters.to_iterable_mapped(_dicts.to_item_at("epoch")) \
        >> _iters.to_list()

    plt.plot(
            list_floats_epochs_training,
            list_floats_loss_training,
            color="#9aa0ab",
            label="training")

    plt.plot(
            list_floats_epochs_evaluation,
            list_floats_loss_evaluation,
            color="#be45ff",
            label="validation")

    plt.legend()

    plt.xlabel("epoch")

    plt.ylabel("loss")

    plt.show()
