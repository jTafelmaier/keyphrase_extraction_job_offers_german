

from lib.unary import _numbers
from lib.unary.main import unary_pure




def to_text(
    int_precision:int = 2):

    @unary_pure()
    def get_text(
        float_value:float):

        text = "{:." \
            + str(int_precision) \
            + "f}"

        return text.format(float_value)

    return get_text


# TODO test further
def to_text_as_percentage(
    int_precision:int = 0):

    @unary_pure()
    def inner(
        float_value:float):

        return float_value \
            >> _numbers.to_number_multiplied_by(100) \
            >> to_text(int_precision)

    return inner

