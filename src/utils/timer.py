# -*- coding: utf-8 -*-
import time
from collections import deque

def format_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

class EmaETA:
    """
    Exponential Moving Average ETA estimator.
    Call update(step_time) every step; use eta(remaining_steps).
    """
    def __init__(self, alpha: float = 0.15, window: int = 20):
        self.alpha = alpha
        self.ema = None
        self.window = deque(maxlen=window)

    def update(self, step_time: float):
        self.window.append(step_time)
        avg = sum(self.window) / len(self.window)
        self.ema = avg if self.ema is None else (self.alpha * avg + (1 - self.alpha) * self.ema)

    def eta(self, remaining_steps: int) -> float:
        if not self.window:
            return 0.0
        return float(self.ema or (sum(self.window) / len(self.window))) * max(0, remaining_steps)

class StepTimer:
    """Context manager for timing code blocks."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
