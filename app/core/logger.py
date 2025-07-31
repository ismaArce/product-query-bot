import logging
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s in %(name)s: %(message)s",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default"],
    },
}


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name (str | None): Logger name. If None, uses the calling module's __name__.
                             Defaults to None.

    Returns:
        logging.Logger: Configured logger instance with application-wide settings.
    """
    if name is None:
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "unknown")
        else:
            name = "unknown"

    logging.config.dictConfig(LOGGING_CONFIG)

    return logging.getLogger(name)


logger = get_logger(__name__)
