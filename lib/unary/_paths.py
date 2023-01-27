

from lib.unary.main import unary




ALIAS_ENCODING_UTF_8 = "utf-8"


def to_file_read(
    id_encoding:str = ALIAS_ENCODING_UTF_8):

    @unary()
    def get_file_read(
        path:str):

        return open(
            file=path,
            mode="r",
            encoding=id_encoding)

    return get_file_read


def to_file_write(
    id_encoding:str = ALIAS_ENCODING_UTF_8):

    @unary()
    def get_file_write(
        path:str):

        return open(
            file=path,
            mode="w",
            encoding=id_encoding)

    return get_file_write


def to_text_file():

    @unary()
    def get_text_file(
        path:str):

        with path >> to_file_read() as file:
            return file.read()

    return get_text_file


def to_path_save_text(
    text:str):

    @unary()
    def save_text(
        path:str):

        with path >> to_file_write() as file:
            file.write(text)

        return path

    return save_text

