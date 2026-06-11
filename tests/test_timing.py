from __future__ import annotations

import logging
import re
import time

from KBzhy.app.core.timing import timed_stage


def test_timed_stage_logs_elapsed_ms(caplog):
    logger = logging.getLogger("tests.perf")

    with caplog.at_level(logging.INFO, logger="tests.perf"):
        with timed_stage(logger, "unit", request_id="rid-1", query="hello"):
            time.sleep(0.001)

    message = caplog.messages[0]
    assert message.startswith("[PERF] stage=unit")
    assert "request_id=rid-1" in message
    assert "query=hello" in message
    assert re.search(r"elapsed_ms=\d+\.\d{2}", message)
