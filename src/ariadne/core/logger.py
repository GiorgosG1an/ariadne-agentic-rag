import logging
import logging.handlers
import queue
import contextvars
import atexit
from pythonjsonlogger import jsonlogger
from typing import Tuple

session_context = contextvars.ContextVar("session_id", default="system")

class ContextFilter(logging.Filter):
    """Injects the `session_id` into every log record automatically."""

    def filter(self, record):
        record.session_id = session_context.get()
        return True


def setup_logger(log_file="workflow.log") ->Tuple[logging.Logger, logging.handlers.QueueListener]:
    """
    Set up a logger with JSON formatting that writes to both file and console.
    Configures a logger for the RAG workflow that outputs structured JSON logs
    to both a file and the console using a queue-based handler system for thread-safe logging.
    
    Returns:

        - tuple: A tuple containing:
            - logger (logging.Logger): The configured logger instance for the RAG workflow.
            - listener (logging.handlers.QueueListener): The queue listener that must be
              started and stopped to manage the logging queue.
    """

    logger = logging.getLogger("RAG_Workflow")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(session_id)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        json_ensure_ascii=False,
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    log_queue = queue.Queue(-1)
    queue_handler = logging.handlers.QueueHandler(log_queue)

    queue_handler.addFilter(ContextFilter())
    logger.addHandler(queue_handler)

    listener = logging.handlers.QueueListener(
        log_queue, file_handler, console_handler, respect_handler_level=True
    )
    listener.start()

    atexit.register(listener.stop)
    return logger, listener
