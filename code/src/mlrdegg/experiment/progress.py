"""Progress reporting for long runs: elapsed time, completed/total fits, percentage and ETA.

The console shows a tqdm bar; every ``log_every`` seconds a line is written to the log and the
same numbers to ``status.json`` so progress can be checked from another terminal.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tqdm import tqdm


def fmt_seconds(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


class Progress:
    def __init__(self, total: int, status_path: Path | None = None, log=None, log_every: float = 60.0):
        self.total, self.done, self.skipped = int(total), 0, 0
        self.omitted = 0
        self.t0 = time.time()
        self.status_path, self.log, self.log_every = status_path, log, log_every
        self._last_log = 0.0
        self.stage = ""
        self.bar = tqdm(total=self.total, unit="fit", dynamic_ncols=True, smoothing=0.05,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} fits "
                                   "[elapsed {elapsed}, ETA {remaining}, {rate_fmt}]")

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.bar.set_description(stage)

    def snapshot(self) -> dict:
        elapsed = time.time() - self.t0
        frac = self.done / self.total if self.total else 1.0
        computed = self.done - self.skipped  # the rate uses newly computed fits only
        eta = elapsed / computed * (self.total - self.done) if computed else float("nan")
        finish = (datetime.now(timezone.utc) + timedelta(seconds=eta)).astimezone().isoformat(
            timespec="seconds") if computed else None
        return {"stage": self.stage, "fits_done": self.done, "fits_total": self.total,
                "fits_omitted": self.omitted,
                "fits_from_cache": self.skipped, "percent": round(100 * frac, 2),
                "elapsed": fmt_seconds(elapsed),
                "eta": fmt_seconds(eta) if computed else "unknown", "expected_finish": finish}

    def skip(self, n: int) -> None:
        """Fits restored from the cache: counted as done, excluded from the rate estimate."""
        self.skipped += n
        self.done += n
        self.bar.reset(total=self.total)  # restart tqdm's clock; count cached fits as initial
        self.bar.initial = self.bar.n = self.bar.last_print_n = self.done
        self.bar.refresh()

    def omit(self, n: int) -> None:
        """Remove unused half tasks or duplicate refits from the initial upper bound."""
        self.omitted += n
        self.total -= n
        self.bar.total = self.total
        self.bar.refresh()

    def update(self, n: int = 1) -> None:
        self.done += n
        self.bar.update(n)
        now = time.time()
        if now - self._last_log >= self.log_every or self.done == self.total:
            self._last_log = now
            snap = self.snapshot()
            if self.log:
                self.log.info("progress %s: %d/%d fits (%.1f%%), elapsed %s, ETA %s", snap["stage"],
                              snap["fits_done"], snap["fits_total"], snap["percent"],
                              snap["elapsed"], snap["eta"])
            self.write_status(state="running")

    def write_status(self, **extra) -> None:
        if self.status_path is None:
            return
        payload = {**self.snapshot(), **extra,
                   "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")}
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.status_path)

    def close(self) -> None:
        self.bar.close()
