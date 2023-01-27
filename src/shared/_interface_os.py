

"""
Utilities to interact with the operating system.
Most file system interaction should be handled via this file.
"""


import os
import typing



def get_path(
    iterable_texts_path:typing.Iterable[str]):

    """
    Return text path by concatenating path sections in "iterable_texts_path".
    - Uses "os.path.join".
    """

    return os.path.join(*iterable_texts_path)

