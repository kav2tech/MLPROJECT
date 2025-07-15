import os
from datetime import datetime
import sys
import logging

LOG_FILE = os.path.join(os.getcwd(), "logs", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def error_message_details(error, error_details):
    """
    Returns a detailed error message as a string.
    """
    _, _, exc_tb = error_details.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_no = "Unknown"
    error_message = (
        f"Error occurred in script: [{file_name}] line number: [{line_no}] error message: [{str(error)}]"
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_details):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details)

    def __str__(self):
        return self.error_message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.error_message!r})"

    def __reduce__(self):
        return (self.__class__, (self.error_message, sys))
    

if __name__ == "__main__":
    try:  
        a = 1/0
    except Exception as e:
        logging.exception("division by zero error")
        raise CustomException(e, sys)
    
#     # print(CustomException(e, sys))  
#     # print(CustomException(e, sys).__reduce__())
#     # print(CustomException(e, sys).__str__())
#     # print(CustomException(e, sys).__repr__())
