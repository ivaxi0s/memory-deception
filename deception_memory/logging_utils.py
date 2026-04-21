from __future__ import annotations

import logging

from rich.logging import RichHandler


def configure_logging(pretty: bool = True) -> None:
    handlers: list[logging.Handler]
    if pretty:
        handlers = [RichHandler(rich_tracebacks=True, markup=False)]
    else:
        handlers = [logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
