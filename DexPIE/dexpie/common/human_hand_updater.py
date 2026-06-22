import threading
import time

from dexpie.common.hand_executor import HAND_MODE_INTERVENTION


class HumanHandUpdater:
    """Feeds MANUS hand targets to the hand executor outside the record loop."""

    def __init__(
        self,
        sample_reader,
        hand_executor,
        is_active_fn,
        update_frequency=120.0,
        sample_timeout=0.25,
    ):
        self.sample_reader = sample_reader
        self.hand_executor = hand_executor
        self.is_active_fn = is_active_fn
        self.update_frequency = float(update_frequency)
        self.sample_timeout = float(sample_timeout)

        if self.update_frequency <= 0:
            raise ValueError("update_frequency must be positive")
        if self.sample_timeout <= 0:
            raise ValueError("sample_timeout must be positive")

        self._stop_event = threading.Event()
        self._error_lock = threading.Lock()
        self._error = None
        self._thread = None
        self._active_since = None

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        with self._error_lock:
            self._error = None
        self._thread = threading.Thread(
            target=self._run_guarded,
            name="dexpie-human-hand-updater",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def raise_if_failed(self):
        with self._error_lock:
            error = self._error
        if error is not None:
            raise error

    def _run_guarded(self):
        try:
            self._loop()
        except Exception as exc:
            with self._error_lock:
                if self._error is None:
                    self._error = RuntimeError(f"human hand updater failed: {exc}")
            self._stop_event.set()

    def _loop(self):
        period = 1.0 / self.update_frequency
        next_tick = time.monotonic()

        while not self._stop_event.is_set():
            if self.is_active_fn():
                if self._active_since is None:
                    self._active_since = time.monotonic()
                self._update_once()
            else:
                self._active_since = None
            next_tick = self._wait_until_next_tick(next_tick, period)

    def _update_once(self):
        sample = self.sample_reader()
        if sample is None:
            if (
                self._active_since is not None
                and time.monotonic() - self._active_since > self.sample_timeout
            ):
                raise RuntimeError(
                    "no MANUS hand sample received within "
                    f"{self.sample_timeout:.3f}s"
                )
            return

        sample_time_ns = int(sample.get("t_host_ns", time.monotonic_ns()))
        sample_age = (time.monotonic_ns() - sample_time_ns) / 1e9
        if sample_age > self.sample_timeout:
            raise RuntimeError(
                f"MANUS hand sample is stale ({sample_age:.3f}s > "
                f"{self.sample_timeout:.3f}s)"
            )

        self.hand_executor.update(
            sample["action"],
            source_mode=HAND_MODE_INTERVENTION,
            target_time_ns=sample_time_ns,
            manus_seq=sample.get("seq"),
            manus_sample_time_ns=sample_time_ns,
        )

    def _wait_until_next_tick(self, previous_tick, period):
        next_tick = previous_tick + period
        delay = next_tick - time.monotonic()
        if delay > 0:
            self._stop_event.wait(delay)
            return next_tick

        missed_periods = int(-delay // period) + 1
        return next_tick + missed_periods * period
