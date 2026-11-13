import logging

from guardstack.logsetup import setup_logging


def test_quiet_filter_survives_a_library_resetting_its_level(caplog):
    setup_logging()
    root = logging.getLogger()
    handler = root.handlers[0]
    logging.getLogger("httpx").setLevel(logging.INFO)          # what something on the cluster did
    rec = logging.LogRecord("httpx", logging.INFO, __file__, 1, "HTTP Request: POST", None, None)
    assert not all(f.filter(rec) for f in handler.filters)
    ours = logging.LogRecord("guardstack.runner", logging.INFO, __file__, 1, "10/20", None, None)
    assert all(f.filter(ours) for f in handler.filters)


def test_logger_filter_survives_root_handler_replacement():
    import io
    setup_logging()
    buf = io.StringIO()
    logging.basicConfig(force=True, stream=buf, level=logging.INFO)   # a library replacing handlers
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpx").info("HTTP Request: POST")
    logging.getLogger("guardstack.runner").info("progress")
    out = buf.getvalue()
    assert "HTTP Request" not in out and "progress" in out
    setup_logging()
