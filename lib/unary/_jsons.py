

# TODO refactor: rename module
# TODO refactor: move some functions

import json
import typing

from lib.unary import _paths
from lib.unary.main import unary




# TODO refactor: rename: does not always return dict
def to_dict_from_json():

    @unary()
    def get_dict_json_from_text(
        text:str):

        return json.loads(text)

    return get_dict_json_from_text


# TODO refactor: perhaps move
# TODO refactor: rename: does not always return dict
def to_dict_json_from_path():

    @unary()
    def get_dict_json_from_path(
        path:str):

        with path >> _paths.to_file_read() as file:
            return json.load(file)

    return get_dict_json_from_path


def to_dict_saved_as_file(
    path:str):

    @unary()
    def save_as_json(
        dictionary:typing.Dict):

        with path >> _paths.to_file_write() as file:
            json.dump(
                obj=dictionary,
                fp=file,
                ensure_ascii=False,
                indent=4)

        return dictionary

    return save_as_json

