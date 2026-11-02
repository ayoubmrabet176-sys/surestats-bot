"""Central logging setup: rotating file handler + console handler."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import settings

_CONFIGURED = False
FMT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    lvl = getattr(logging, (level or settings.log_level), logging.INFO)
    root = logging.getLogger("surestats")
    root.setLevel(lvl)
    root.propagate = False

    fmt = logging.Formatter(FMT, datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    log_file = Path(settings.log_dir) / "surestats.log"
    try:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        pass

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"surestats.{name}")
