"""In-process guard models are loaded once per process and shared by the runner's threads."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from guardstack.layers import _adapter_common as C
from guardstack.layers.safety_classifiers import HFPromptInjectionClassifier
from guardstack.types import Stage

LOADS = []


class _FakePI(HFPromptInjectionClassifier):
    def _load(self):
        LOADS.append(threading.get_ident())
        time.sleep(0.05)          # widen the race window
        model = SimpleNamespace(config=SimpleNamespace(id2label={0: "SAFE", 1: "INJECTION"}))
        return object(), model, "cpu"


def test_sixteen_threads_load_once_and_share():
    LOADS.clear()
    layers = [_FakePI(stage=Stage.TOOL_OUTPUT) for _ in range(16)]
    ts = [threading.Thread(target=l.warmup) for l in layers]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(LOADS) == 1
    assert len({id(l._model) for l in layers}) == 1
    assert len({id(x._infer_lock) for x in layers}) == 1 and layers[0]._infer_lock is not C.NOLOCK


def test_distinct_models_and_loaders_are_not_conflated():
    a = C.shared_load(object(), lambda: "x", "model-a", None)
    b = C.shared_load(object(), lambda: "y", "model-a", None)   # another loader (e.g. a test fake)
    c = C.shared_load(object(), lambda: "z", "model-b", None)
    assert a[0] == "x" and b[0] == "y" and c[0] == "z"
