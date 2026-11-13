"""Logging for the job scripts: our progress lines at INFO, third-party chatter at WARNING or above.

NeMo Guardrails logs every internal event at INFO (the first shadow_all smoke wrote 4 MB of it for
15 sessions, which would be gigabytes for a full run), presidio repeats a warning per recognizer
each time an analyzer is built, and httpx logs every request."""
from __future__ import annotations

import logging

QUIET = {
    "nemoguardrails": logging.WARNING,
    "presidio-analyzer": logging.ERROR,     # "Recognizer not added to registry ..." x17 per analyzer
    "presidio_analyzer": logging.ERROR,
    "httpx": logging.WARNING,
    "httpx2": logging.WARNING,         # the HTTP client the openai SDK on the cluster uses (found Nov 9 via [name])
    "httpcore": logging.WARNING,
    "httpcore2": logging.WARNING,
    "openai": logging.WARNING,
    "urllib3": logging.WARNING,
    "gliner": logging.WARNING,
    "transformers": logging.WARNING,
    "sentence_transformers": logging.WARNING,
}


class _QuietFilter(logging.Filter):
    """Drops records below the QUIET threshold of their logger family. A handler filter, so it still
    holds when a library later resets its own logger level (on the cluster something set httpx back to INFO:
    22k "HTTP Request" lines per static-phase job despite the setLevel below)."""

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        for prefix, lvl in QUIET.items():
            if name == prefix or name.startswith(prefix + "."):
                return record.levelno >= lvl
        return True


def setup_logging(level: int = logging.INFO) -> None:
    # %(name)s: shows which logger the remaining httpx-style lines on the cluster really come from
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    for name, lvl in QUIET.items():
        lg = logging.getLogger(name)
        lg.setLevel(lvl)
        # also on the logger itself: a logger filter runs before any handler, so it holds even if a
        # library resets the level or replaces the root handlers (the handler filter alone did not
        # stop httpx's lines on the cluster)
        if not any(isinstance(f, _QuietFilter) for f in lg.filters):
            lg.addFilter(_QuietFilter())
    for h in logging.getLogger().handlers:
        if not any(isinstance(f, _QuietFilter) for f in h.filters):
            h.addFilter(_QuietFilter())


def describe_logging() -> str:
    """One line on the live logging setup, logged at the end of a run to see what changed it."""
    root = logging.getLogger()
    hx = logging.getLogger("httpx")
    return (f"logging: root level={root.level} handlers={[type(h).__name__ for h in root.handlers]} "
            f"handler_filters={[len(h.filters) for h in root.handlers]} httpx level={hx.level} "
            f"httpx handlers={[type(h).__name__ for h in hx.handlers]} httpx filters={len(hx.filters)} "
            f"propagate={hx.propagate}")
