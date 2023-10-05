import logging


def setup_logger():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname).1s: %(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger("toyflow")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()
