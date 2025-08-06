import logging
import os
from datetime import datetime
from src.exception import CustomException
import sys

# Configure logging
LOG_FILE = os.path.join(os.getcwd(), "logs", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_info(message: str):
    logging.info(message)   

def log_error(message: str):
    logging.error(message)

def log_warning(message: str):
    logging.warning(message)  

# ✅ This should be OUTSIDE the function!
if __name__ == "__main__":
    
    log_info("This is an INFO message.")
    log_warning("This is a WARNING message.")
    log_error("This is an ERROR message.")
    print(f"Logging to: {LOG_FILE}")

    
        
# This code sets up a custom logger that logs messages to a file with timestamps.
# The log file is created in a 'logs' directory with the current date and time in the filename.
# The logger supports different log levels: info, warning, and error.   
# You can use the `log_info`, `log_warning`, and `log_error` functions to log messages at different levels.
# The log messages will be formatted with the timestamp, log level, and message content.
# The log file is created in the current working directory under a 'logs' subdirectory.
# If the 'logs' directory does not exist, it will be created automatically.