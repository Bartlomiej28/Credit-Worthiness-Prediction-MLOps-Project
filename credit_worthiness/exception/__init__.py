import sys


def error_message_detail(error, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return f"Error message [{error}]"

    while exc_tb.tb_next is not None:
        exc_tb = exc_tb.tb_next

    file_name = exc_tb.tb_frame.f_code.co_filename
    return "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )


class CreditWorthinessException(Exception):
    def __init__(self, error_message, error_detail):
        """
        :param error_message: error message in string format
        :param error_detail: the sys module, used to reach the active traceback
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
