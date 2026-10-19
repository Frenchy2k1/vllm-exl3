"""Per-layer routed-expert routing-frequency histogram (off by default).

Why this exists
---------------
The ATS alias set is chosen today by *layer*, not by *traffic*: layers 38-39
and all three MTP layers are aliased in their entirety, so a decode step pays
an NVMe/host-mapped read for every one of the ~40 distinct experts it touches
in those layers (``lanes/non-moe-slice`` §4.3: 130 ms/step, 44 % of MoE time).
Choosing the alias set by measured routing frequency needs the frequency.

What it does
------------
One preallocated ``int64[n_rows, n_experts]`` device buffer plus two
``int64[n_rows]`` counters.  Every MoE forward does

    counts[row].index_add_(0, topk_ids.reshape(-1), ones[:topk_ids.numel()])
    calls[row].add_(1)
    rows[row].add_(tokens)

i.e. three tiny kernels, no allocation, no host sync, no branch on data.  All
three tensors are allocated once per device, so a captured CUDA graph keeps
fixed addresses and every replay accumulates normally (this is the only
mechanism that works under ``cudagraph_mode=FULL_*``: replay does not re-run
Python, so any host-side counter would undercount by the whole run).

``counts`` is a **route** histogram, not a step histogram: ``p_e =
counts[row, e] / rows[row]`` is the marginal router probability of expert ``e``
in layer ``row``.  Prefill and decode are pooled on purpose -- both are draws
from the same router, and the alias traffic model only needs ``p_e``.

Dumping (never inside the forward)
----------------------------------
Three triggers, none of which needs a host sync in the forward:

* ``atexit`` -- clean shutdown;
* ``SIGUSR1`` -- handler only sets a :class:`threading.Event` (no CUDA in a
  signal handler); a daemon thread does the copy;
* a **trigger file** polled by that same daemon thread every 2 s
  (``$VLLM_EXL3_ROUTE_STATS_TRIGGER``, default ``<dir>/dump.now``) -- this is
  the primary one, because it also works when the container is stopped with
  ``docker stop`` (SIGTERM -> atexit usually runs, but SIGKILL does not) and it
  lets a calibration script flush *between* content types without restarting
  the engine.

Optionally ``VLLM_EXL3_ROUTE_STATS_DUMP_SEC`` adds a periodic dump.

The dump blocks the engine for `cuda_synchronize + R*E*8 bytes` (43*384*8 =
132 KiB) ~1 ms.  That is acceptable off the measured path and is why the
default is trigger-only.

Env
---
``VLLM_EXL3_ROUTE_STATS=1``            master switch
``VLLM_EXL3_ROUTE_STATS_DIR``          output dir (default ``/tmp``)
``VLLM_EXL3_ROUTE_STATS_ROWS``         rows in the buffer (default 48)
``VLLM_EXL3_ROUTE_STATS_EXPERTS``      columns (default 384; MTP rows use 128)
``VLLM_EXL3_ROUTE_STATS_MTP_OFFSET``   row of ``mtp.0`` (default 40)
``VLLM_EXL3_ROUTE_STATS_TRIGGER``      trigger-file path
``VLLM_EXL3_ROUTE_STATS_DUMP_SEC``     periodic dump period, 0 = off
``VLLM_EXL3_ROUTE_STATS_LIGHT=1``      skip the ``calls``/``rows`` counters
                                       (1 kernel/layer instead of 3)
``VLLM_EXL3_ROUTE_STATS_RANK0_ONLY=0`` count on every rank (default: only
                                       ``RANK``/``LOCAL_RANK`` 0, to avoid
                                       R ranks writing the same file)
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import signal
import threading
import time

logger = logging.getLogger(__name__)

_LAYER_RE = re.compile(r"(?:^|\.)((?:layers|mtp)\.(\d+))(?:\.|$)")

_STATE: dict[int, dict] = {}
_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None
_WAKE = threading.Event()
_ENABLED: bool | None = None
_N_ROWS = 48
_N_EXP = 384
_MTP_OFFSET = 40
_OUT_DIR = "/tmp"
_LIGHT = False
_TRIGGER = ""
_DUMP_SEC = 0.0
_DUMPS = 0


def _env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False")


def enabled() -> bool:
    global _ENABLED, _N_ROWS, _N_EXP, _MTP_OFFSET, _OUT_DIR, _LIGHT
    global _TRIGGER, _DUMP_SEC
    if _ENABLED is not None:
        return _ENABLED
    _ENABLED = _env_on("VLLM_EXL3_ROUTE_STATS")
    if not _ENABLED:
        return False
    for attr, var, conv in (
        ("_N_ROWS", "VLLM_EXL3_ROUTE_STATS_ROWS", int),
        ("_N_EXP", "VLLM_EXL3_ROUTE_STATS_EXPERTS", int),
        ("_MTP_OFFSET", "VLLM_EXL3_ROUTE_STATS_MTP_OFFSET", int),
        ("_DUMP_SEC", "VLLM_EXL3_ROUTE_STATS_DUMP_SEC", float),
    ):
        try:
            val = conv(os.environ.get(var, ""))
        except (TypeError, ValueError):
            continue
        globals()[attr] = val
    _OUT_DIR = os.environ.get("VLLM_EXL3_ROUTE_STATS_DIR", "/tmp")
    _TRIGGER = os.environ.get(
        "VLLM_EXL3_ROUTE_STATS_TRIGGER", os.path.join(_OUT_DIR, "dump.now")
    )
    _LIGHT = _env_on("VLLM_EXL3_ROUTE_STATS_LIGHT")
    try:
        os.makedirs(_OUT_DIR, exist_ok=True)
    except OSError:
        pass
    _start_watchdog()
    logger.info(
        "EXL3 route stats: ON rows=%d experts=%d mtp_offset=%d dir=%s light=%s",
        _N_ROWS,
        _N_EXP,
        _MTP_OFFSET,
        _OUT_DIR,
        _LIGHT,
    )
    return True


def _row_of(layer) -> int | None:
    """Histogram row for a MoE module, or None if it is not a routed block.

    ``model.layers.38`` -> 38, ``model.layers.40`` (DSpark draft block built on
    top of the 40-layer stack) -> 40, ``mtp.1`` -> 41.
    """
    name = str(
        getattr(layer, "layer_name", None)
        or getattr(layer, "prefix", None)
        or ""
    )
    m = _LAYER_RE.search(name)
    if not m:
        return None
    idx = int(m.group(2))
    if m.group(1).startswith("mtp."):
        idx += _MTP_OFFSET
    return idx if 0 <= idx < _N_ROWS else None


def _state(device):
    import torch

    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    st = _STATE.get(idx)
    if st is not None:
        return st
    with _LOCK:
        st = _STATE.get(idx)
        if st is not None:
            return st
        st = {
            "counts": torch.zeros((_N_ROWS, _N_EXP), dtype=torch.int64, device=dev),
            "calls": torch.zeros((_N_ROWS,), dtype=torch.int64, device=dev),
            "rows": torch.zeros((_N_ROWS,), dtype=torch.int64, device=dev),
            # Static ``ones`` source for index_add_: slicing a preallocated
            # tensor keeps the address fixed under graph capture.
            "ones": torch.ones((1 << 15,), dtype=torch.int64, device=dev),
        }
        _STATE[idx] = st
        return st


def record(layer, topk_ids) -> None:
    """Accumulate one MoE forward's route counts.  Never syncs, never allocates.

    ``topk_ids`` is ``[tokens, top_k]`` in any int dtype; ``index_add_`` takes
    int32 or int64 indices, so no cast (and no allocation) is needed.
    """
    if not enabled():
        return
    try:
        import torch

        if topk_ids is None or not torch.is_tensor(topk_ids):
            return
        row = _row_of(layer)
        if row is None:
            return
        st = _state(topk_ids.device)
        flat = topk_ids.reshape(-1)
        n = int(flat.numel())
        if n == 0:
            return
        if n > st["ones"].numel():
            with _LOCK:
                st["ones"] = torch.ones(
                    (n,), dtype=torch.int64, device=topk_ids.device
                )
            # Re-allocating mid-capture would move the address; log loudly.
            logger.warning(
                "EXL3 route stats: ones buffer grown to %d mid-run; if this "
                "happens after graph capture the counts are suspect",
                n,
            )
        # Ids come from the router, but the profile_run / dummy path can feed
        # uninitialised garbage (observed: device-side assert dstIndex < 384 at
        # boot).  Real ids are < _N_EXP, so remainder is the identity for them.
        if flat.dtype != torch.int64:
            flat = flat.to(torch.int64)
        flat = flat.remainder(_N_EXP)
        st["counts"][row].index_add_(0, flat, st["ones"][:n])
        if not _LIGHT:
            st["calls"][row].add_(1)
            st["rows"][row].add_(int(topk_ids.shape[0]))
    except Exception as exc:  # instrumentation must never break a forward
        global _ENABLED
        _ENABLED = False
        logger.warning("EXL3 route stats disabled after error: %r", exc)


def _dump_once(tag: str = "") -> str | None:
    global _DUMPS
    if not _STATE:
        return None
    import numpy as np
    import torch

    out = {}
    calls = rows = None
    with _LOCK:
        for idx, st in _STATE.items():
            torch.cuda.synchronize(idx)
            out[f"counts_{idx}"] = st["counts"].cpu().numpy().copy()
            out[f"calls_{idx}"] = st["calls"].cpu().numpy().copy()
            out[f"rows_{idx}"] = st["rows"].cpu().numpy().copy()
        meta = {
            "n_rows": _N_ROWS,
            "n_experts": _N_EXP,
            "mtp_offset": _MTP_OFFSET,
            "light": _LIGHT,
            "dumps": _DUMPS,
            "tag": tag,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": "counts[row,e] = token-routes to expert e; p_e = counts/rows",
        }
        out["meta"] = np.array(json.dumps(meta))
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:64] if tag else ""
        name = f"route_stats_{safe}.npz" if safe else "route_stats.npz"
        path = os.path.join(_OUT_DIR, name)
        tmp = path + ".tmp"
        np.savez(tmp, **out)
        os.replace(tmp, path)
        _DUMPS += 1
    logger.info("EXL3 route stats: dumped %s", path)
    return path


def dump(tag: str = "") -> str | None:
    """Public dump entry point (safe to call from any non-forward thread)."""
    try:
        return _dump_once(tag)
    except Exception as exc:  # pragma: no cover
        logger.warning("EXL3 route stats: dump failed: %r", exc)
        return None


def _watchdog() -> None:
    last = 0.0
    while True:
        try:
            _WAKE.wait(timeout=2.0)
            _WAKE.clear()
            if _TRIGGER and os.path.exists(_TRIGGER):
                tag = ""
                try:
                    with open(_TRIGGER, encoding="utf-8") as fh:
                        tag = fh.read().strip()
                except OSError:
                    pass
                try:
                    os.unlink(_TRIGGER)
                except OSError:
                    pass
                dump(tag)
            if _DUMP_SEC > 0 and time.time() - last >= _DUMP_SEC:
                last = time.time()
                dump()
        except Exception:  # pragma: no cover - never kill the thread
            time.sleep(1.0)


def _start_watchdog() -> None:
    global _THREAD
    if _THREAD is not None:
        return
    _THREAD = threading.Thread(target=_watchdog, name="exl3-route-stats", daemon=True)
    _THREAD.start()

    def _on_usr1(_sig, _frm):
        _WAKE.set()

    try:
        signal.signal(signal.SIGUSR1, _on_usr1)
    except (ValueError, OSError):
        pass
    atexit.register(lambda: dump("atexit"))


def rank_should_count() -> bool:
    """True on rank 0 (or when ``VLLM_EXL3_ROUTE_STATS_RANK0_ONLY=0``)."""
    if not _env_on("VLLM_EXL3_ROUTE_STATS_RANK0_ONLY", "1"):
        return True
    for var in ("RANK", "LOCAL_RANK", "VLLM_DP_RANK"):
        v = os.environ.get(var)
        if v not in (None, ""):
            try:
                return int(v) == 0
            except ValueError:
                continue
    return True