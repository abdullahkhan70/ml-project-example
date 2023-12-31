import sys


def error_message_detail(message, error_detail:sys):
    _, _, exe_tb = error_detail.exc_info()  
    file_name = exe_tb.tb_frame.f_code.co_filename
    line_no = exe_tb.tb_lineno
    error_message = f"Error occured in script name: {file_name} \
        \n Line Number: {line_no}\n Error Message: {str(message)}"

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys) -> None:
        super().__init__()
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
        