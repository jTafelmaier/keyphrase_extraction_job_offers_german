

from __future__ import annotations
from typing import TypeVar, Generic
import typing




Type_input = TypeVar("Type_input")
Type_output = TypeVar("Type_output")
Type_output_other = TypeVar("Type_output_other")


class Function_unary(Generic[Type_input, Type_output]):

    def __init__(
        self:Function_unary,
        function:typing.Callable[[Type_input], Type_output],
        bool_is_pure:bool):

        self.function = function
        self.bool_is_pure = bool_is_pure


    def __or__(
        self:Function_unary,
        function_other:Function_unary[Type_output, Type_output_other]):

        assert isinstance(
            function_other,
            Function_unary)

        @unary(bool_is_pure=self.bool_is_pure and function_other.bool_is_pure)
        def get_result_compose(
            item:typing.Any):

            return item \
                >> self \
                >> function_other

        return get_result_compose


    def __rshift__( # NOTE perhaps only necessary for using Function_unary as input (TODO verify)
        self:Function_unary,
        function_other:Function_unary):

        assert isinstance(
            function_other,
            Function_unary)

        return function_other(self)


    def __rrshift__(
        self:Function_unary,
        item_argument:Type_input):

        return self.function(item_argument)


    def __call__(
        self:Function_unary,
        item_argument:Type_input):

        return self.function(item_argument)


def unary(
    bool_is_pure:bool = False):

    """
    Designate this function as a unary function.
    This enables function chaining (via ">>") and function composition (via "|").
    Pure functions may be executed concurrently.
    """

    def get_function_unary(
        function:typing.Callable[[Type_input], Type_output]):

        return Function_unary[Type_input, Type_output](
            function=function,
            bool_is_pure=bool_is_pure)

    return get_function_unary


def unary_pure():
    return unary(bool_is_pure=True)


# function chaining should be included
# the symbol ">>" should be the most used symbol and have a simple meaning
# the symbol ">>" should be the leftmost symbol in each row
# when a function chain is called on an item, it should be positioned before the function chain
# operators alone (not variable types) need to be sufficient in explaining flow

