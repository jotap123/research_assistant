import logging
import os

import colorlog

LOGLEVEL = os.getenv("ATENA_LOGLEVEL", "INFO")


def init():
    log_format = (
        "%(asctime)s - "
        "%(name)s - "
        "%(funcName)s - "
        "%(levelname)s - "
        "%(message)s"
    )
    colorlog_format = f"%(log_color)s{log_format}"
    colorlog.basicConfig(format=colorlog_format, level=getattr(logging, LOGLEVEL))
    logging.getLogger("azure").setLevel(logging.WARN)
