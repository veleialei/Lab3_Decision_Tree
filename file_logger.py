#file_log.txt
from pathlib import Path
from logger import logger

# logger.log(0, 'Starting logger (type = ' + logger_type + ') at log level ' + str(log_level) + '.')
# logger.log(1, 'Important message.')
# logger.log(2, 'Less important message.')
# logger.log(3, 'Not important message.')
# logger.log(0, 'Ending logger.')

class file_logger(logger):

    def __init__(self, log_level):
        self.__log_level__ = log_level
        my_file = Path("file_log.txt")
        if my_file.is_file():
            self.filename = input("please give me a name: ") + ".txt"
        else:
            self.filename = "file_log.txt"

    def log(self, log_level, message):
        fo = open(self.filename, "a")
        fo.write(str(log_level) + ": " + message + "\n")
        fo.close()
