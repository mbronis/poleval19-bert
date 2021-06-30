import logging


logging.basicConfig(filename='log.log', level=logging.INFO)


class Logger:
    def __init__(self, name: str) -> None:
        self.logger = logging.getLogger(name)

    def log(self, message: str) -> None:
        self.logger.info(message)
