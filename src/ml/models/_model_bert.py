

import typing
import datetime

import pandas as pd
import torch
import numpy as np
import transformers
from sklearn import metrics as _sklearn_metrics
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from lib.unary import _items
from lib.unary import _iters
from lib.unary import _tuples
from lib.unary.bs4 import _htmls
from lib.unary.main import unary

from src.ml import _ml_retrieval




# TODO change: perhaps change
transformers.logging.set_verbosity_error()

PATH_MODEL = "resources/data_ml/data_BERT/dict_state_model.pt"

PATH_TOKENIZER = "resources/data_ml/data_BERT/tokenizer/"

NAME_MODEL_BERT = "bert-base-german-cased"

SIZE_BATCHES_TRAIN = 6


# TODO refactor: perhaps move
def get_device():

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_iterable_data_tokenized_for_bert(
    tokenizer:transformers.BertTokenizer):

    # TODO investigate: return type: "BatchEncoding"
    @unary()
    def get_dict_tokenized(
        html:str):

        ALIAS_TYPE_TENSOR_PYTORCH = "pt"

        text = html \
            >> _htmls.to_text_content()

        # TODO try to replace "__call__" by appropriate method
        return tokenizer(
            text=text,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors=ALIAS_TYPE_TENSOR_PYTORCH)

    @unary()
    def inner(
        iterable_htmls:typing.Iterable[str]):

        return iterable_htmls \
            >> _iters.to_iterable_mapped(get_dict_tokenized)

    return inner


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataframe_data:pd.DataFrame):

        tokenizer = transformers.BertTokenizer.from_pretrained(NAME_MODEL_BERT)

        self.dataframe_labels = dataframe_data \
            .iloc[:, 5:]

        # TODO refactor: perhaps convert to array
        self.tuple_dicts_tokenized = dataframe_data \
            .iloc[:, 3] \
            >> to_iterable_data_tokenized_for_bert(tokenizer) \
            >> _iters.to_tuple()

        # TODO refactor: use constant
        tokenizer.save_pretrained("resources/data_ml/data_BERT/tokenizer/temp/")


    def __len__(self):
        return len(self.tuple_dicts_tokenized)


    def __getitem__(
        self,
        idx):

        batch_texts = self.tuple_dicts_tokenized \
            [idx]

        batch_y = np.array(self.dataframe_labels \
            .iloc[idx, :])

        return (
            batch_texts,
            batch_y)


# TODO refactor: try huggingface implementation of BERT
class Model(nn.Module):

    def __init__(
        self,
        probability_dropout:float = 0.5):

        super(
            Model,
            self) \
            .__init__()

        # TODO implement: perhaps try Bert large
        self.layer_bert = transformers.BertModel.from_pretrained(NAME_MODEL_BERT)

        # TODO implement: try different values as well as more layers
        number_neurons_middle = 200

        number_neurons_last = _ml_retrieval.NUMBER_CATEGORIES_PROFESSIONS \
            + _ml_retrieval.NUMBER_CATEGORIES_MODES_EMPLOYMENT

        # TODO refactor
        self.layer_fully_connected_last = torch.nn.Sequential(
            nn.Linear(
                in_features=768,
                out_features=number_neurons_middle),
            nn.ReLU(),
            nn.Dropout(probability_dropout),
            nn.Linear(
                in_features=number_neurons_middle,
                out_features=number_neurons_last))

    # TODO refactor: rename arguments
    def forward(
        self,
        input_id,
        mask):

        _, \
        tensor_embedding_cls = self.layer_bert.forward(
            input_ids=input_id,
            attention_mask=mask,
            return_dict=False)

        return self.layer_fully_connected_last.forward(tensor_embedding_cls)


def get_float_f1(
    tensor_predictions:torch.Tensor,
    tensor_truth:torch.Tensor):

    tensor_predictions_binary = tensor_predictions \
        >= 0.5

    return _sklearn_metrics.f1_score(
        y_true=tensor_truth.cpu(),
        y_pred=tensor_predictions_binary.cpu(),
        average="micro",
        zero_division=1)


# TODO investigate if type annotation for "model" can be more general
def get_tensor_output_forward(
    model:Model,
    batch_encoding:transformers.tokenization_utils_base.BatchEncoding,
    device:torch.device):

    # TODO refactor
    tensor_mask = batch_encoding \
        ["attention_mask"] \
        .to(device)

    tensor_ids_input = batch_encoding \
        ["input_ids"] \
        .squeeze(1) \
        .to(device)

    return model.forward(
        input_id=tensor_ids_input,
        mask=tensor_mask)


def get_tuple_metrics_over_dataframe(
    model:Model,
    dataframe_data:pd.DataFrame):

    dataloader = torch.utils.data.DataLoader(
        dataset=Dataset(dataframe_data),
        batch_size=1)

    device = get_device()

    # TODO refactor
    model = model \
        .to(device)

    @unary(bool_is_pure=False)
    def get_tuple_metrics_batch(
        tuple_batch:typing.Tuple):

        # TODO refactor: perhaps rename "batch_encoding"
        batch_encoding, \
        tensor_labels = tuple_batch

        tensor_labels = tensor_labels \
            .to(device) \
            .float()

        tensor_predictions = get_tensor_output_forward(
            model=model,
            batch_encoding=batch_encoding,
            device=device)

        tensor_predictions_binary = (torch.sigmoid(tensor_predictions) \
            >= 0.5) \
            .cpu()

        tensor_labels_cpu = tensor_labels \
            .cpu()

        float_recall = _sklearn_metrics.recall_score(
                y_true=tensor_labels_cpu,
                y_pred=tensor_predictions_binary,
                average="micro",
                zero_division=0.0)

        float_precision = _sklearn_metrics.precision_score(
                y_true=tensor_labels_cpu,
                y_pred=tensor_predictions_binary,
                average="micro",
                zero_division=0.0)

        float_f1 = _sklearn_metrics.f1_score(
                y_true=tensor_labels_cpu,
                y_pred=tensor_predictions_binary,
                average="micro",
                zero_division=0.0)

        return (
            float_recall,
            float_precision,
            float_f1)

    with torch.no_grad():
        return tuple(pd.DataFrame(dataloader \
            >> _iters.to_iterable_mapped(get_tuple_metrics_batch) \
            >> _iters.to_list()) \
            .mean(
                axis=0)
            .to_list())


def train(
    model:Model,
    dataframe_data_training:pd.DataFrame,
    dataframe_data_validation:pd.DataFrame,
    float_learning_rate:float,
    size_batches:int,
    count_epochs:int):

    # TODO refactor: shuffle probably not necessary
    dataloader_training = torch.utils.data.DataLoader(
        dataset=Dataset(dataframe_data_training),
        batch_size=size_batches,
        shuffle=True)

    device = get_device()

    # TODO refactor
    model = model \
        .to(device)

    # TODO implement: try nn.MultiLabelSoftMarginLoss
    criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(10.0)) \
        .to(device)

    optimizer = Adam(
        params=model.parameters(),
        lr=float_learning_rate)

    for epoch_num in range(count_epochs):

        float_f1_total_train = 0.0

        for batch_encoding, tensor_labels in tqdm(dataloader_training):

            tensor_labels = tensor_labels \
                .to(device) \
                .float()

            tensor_output = get_tensor_output_forward(
                model=model,
                batch_encoding=batch_encoding,
                device=device)

            batch_loss = criterion.forward(
                input=tensor_output,
                target=tensor_labels)

            float_f1_total_train += get_float_f1(
                tensor_predictions=tensor_output,
                tensor_truth=tensor_labels)

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        float_recall_validation, \
        float_precision_validation, \
        float_f1_validation = get_tuple_metrics_over_dataframe(
                model=model,
                dataframe_data=dataframe_data_validation)

        # TODO refactor
        print(
            f'Epochs: {epoch_num + 1} | F1 Train: {float_f1_total_train / len(dataloader_training): .3f} | F1 Validation: {float_f1_validation: .3f}')

        text_time = datetime.datetime \
            .now() \
            .strftime("%Y-%m-%dT%H_%M_%S")

        # TODO implement: create new function to convert the state dict to the cpu version. To do so: load state dict with CUDA, move model to cpu with model.cpu() and then save state dict again.

        torch.save(
            obj=model.state_dict(),
            f="resources/data_ml/data_BERT/dict_state_model" \
                + text_time \
                + ".pt")


def evaluate(
    model:Model,
    dataframe_data_testing:pd.DataFrame):

    float_recall, \
    float_precision, \
    float_f1 = get_tuple_metrics_over_dataframe(
        model=model,
        dataframe_data=dataframe_data_testing)

    # TODO refactor
    print(f'Recall Test: {float_recall: .3f}\nPrecision Test: {float_precision: .3f}\nF1 Test: {float_f1: .3f}')


# TODO refactor: perhaps includes some duplicate code
# TODO refactor: perhaps rename
def get_list_pairs_probabilites_categories(
    model:Model,
    tokenizer:transformers.BertTokenizer,
    device:torch.device,
    html:str):

    batch_encoding = html \
        >> _items.to_iterable_singleton() \
        >> to_iterable_data_tokenized_for_bert(tokenizer) \
        >> _iters.to_item_first(
            do_allow_default=False)

    # TODO refactor
    return torch.sigmoid(get_tensor_output_forward(
            model=model,
            batch_encoding=batch_encoding,
            device=device) \
        .flatten()) \
        .tolist() \
        >> _iters.to_iterable_enumerated() \
        >> _iters.to_list_sorted(
            function_get_key=_tuples.to_item_last(),
            do_sort_descending=True)


def train_new_model():

    np.random.seed(112)

    torch.manual_seed(112)

    dataframe_data_training = _ml_retrieval.get_dataframe_data_partition(_ml_retrieval.PATH_CSV_DATA_TRAINING)

    dataframe_data_validation = _ml_retrieval.get_dataframe_data_partition(_ml_retrieval.PATH_CSV_DATA_VALIDATION)

    dataframe_data_testing = _ml_retrieval.get_dataframe_data_partition(_ml_retrieval.PATH_CSV_DATA_TESTING)

    model = Model()

    # model.load_state_dict(torch.load(PATH_MODEL))

    # TODO implement: perhaps try a higher learning rate
    train(
        model=model,
        dataframe_data_training=dataframe_data_training,
        dataframe_data_validation=dataframe_data_validation,
        float_learning_rate=1e-5,
        size_batches=SIZE_BATCHES_TRAIN,
        count_epochs=20)

    evaluate(
        model=model,
        dataframe_data_testing=dataframe_data_testing)

