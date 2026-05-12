"""
Logging utilities for the Taxapp API.

Provides:
- `Colors`: ANSI color helpers (auto-disabled when stdout is not a TTY).
- `ColorFormatter`: colorized `logging.Formatter`.
- `setup_logging`: configure the root logger + tame noisy third-party loggers.
- `install_access_log_middleware`: register an HTTP access-log middleware on a FastAPI app.
"""

import logging
import os
import sys
import time

from fastapi import FastAPI, Request


class Colors:
    """ANSI escape codes. Auto-disabled when stdout is not a TTY."""

    _enabled = sys.stdout.isatty() and os.getenv("NO_COLOR") is None

    RESET = "\033[0m" if _enabled else ""
    BOLD = "\033[1m" if _enabled else ""
    DIM = "\033[2m" if _enabled else ""
    GRAY = "\033[90m" if _enabled else ""
    RED = "\033[31m" if _enabled else ""
    GREEN = "\033[32m" if _enabled else ""
    YELLOW = "\033[33m" if _enabled else ""
    BLUE = "\033[34m" if _enabled else ""
    MAGENTA = "\033[35m" if _enabled else ""
    CYAN = "\033[36m" if _enabled else ""
    WHITE = "\033[37m" if _enabled else ""


class ColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: Colors.GRAY,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
        ts_str = f"{Colors.GRAY}{ts}{Colors.RESET}"
        level_str = f"{level_color}{record.levelname:<7}{Colors.RESET}"
        name_str = f"{Colors.CYAN}{record.name}{Colors.RESET}"
        loc_str = f"{Colors.DIM}{record.filename}:{record.lineno}{Colors.RESET}"
        msg = record.getMessage()
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"
        return (
            f"{ts_str} {Colors.DIM}|{Colors.RESET} {level_str} {Colors.DIM}|{Colors.RESET} "
            f"{name_str} {Colors.DIM}|{Colors.RESET} {loc_str} {Colors.DIM}|{Colors.RESET} {msg}"
        )


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    # Route uvicorn's loggers through our color formatter and silence its
    # default access log since we emit our own (richer) access line below.
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers = [handler]
        lg.propagate = False
    logging.getLogger("uvicorn.access").disabled = True

    # Third-party libraries that emit INFO chatter on every request / startup.
    # Bump them to WARNING so only real problems show up.
    for noisy in (
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
        "chromadb",
        "httpx",
        "httpcore",
        "anthropic",
        "urllib3",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def install_access_log_middleware(app: FastAPI, logger: logging.Logger) -> None:
    """Register an HTTP access-log middleware on the given FastAPI app."""

    @app.middleware("http")
    async def access_log_middleware(request: Request, call_next):
        """Log every HTTP request with timestamp, IP, method, path, status, latency."""
        start = time.perf_counter()

        # Prefer X-Forwarded-For when behind a proxy, else direct peer.
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        elif request.client:
            client_ip = request.client.host
        else:
            client_ip = "unknown"

        method = request.method
        path = request.url.path
        query = request.url.query
        endpoint = f"{path}?{query}" if query else path

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                f"{Colors.BOLD}{method}{Colors.RESET} "
                f"{Colors.BLUE}{endpoint}{Colors.RESET} "
                f"{Colors.DIM}from{Colors.RESET} {Colors.CYAN}{client_ip}{Colors.RESET} "
                f"{Colors.RED}500{Colors.RESET} "
                f"{Colors.DIM}in{Colors.RESET} {elapsed_ms:.1f}ms"
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        status = response.status_code
        if status >= 500:
            status_color = Colors.RED
        elif status >= 400:
            status_color = Colors.YELLOW
        elif status >= 300:
            status_color = Colors.CYAN
        else:
            status_color = Colors.GREEN

        logger.info(
            f"{Colors.BOLD}{method}{Colors.RESET} "
            f"{Colors.BLUE}{endpoint}{Colors.RESET} "
            f"{Colors.DIM}from{Colors.RESET} {Colors.CYAN}{client_ip}{Colors.RESET} "
            f"{status_color}{status}{Colors.RESET} "
            f"{Colors.DIM}in{Colors.RESET} {elapsed_ms:.1f}ms"
        )
        return response
