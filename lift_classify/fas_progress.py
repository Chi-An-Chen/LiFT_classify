"""Batch progress rendering, independent of optimization and metric calculation."""

import time

from rich.progress import (BarColumn, Progress, TextColumn, TimeRemainingColumn)
from rich.table import Column


def make_progress(*, console=None):
    return Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed:.0f}/{task.total:.0f}"),
        TextColumn("{task.fields[stats]} · {task.fields[speed]} img/s · peak={task.fields[peak]} GiB",
                   table_column=Column(no_wrap=True)),
        TextColumn("ETA"), TimeRemainingColumn(),
        refresh_per_second=4,
        console=console,
    )


class BatchProgress:
    def __init__(self, progress, description, total, *, validation=False):
        self.progress = progress
        self.validation = validation
        self.task = progress.add_task(description, total=total, start=False,
                                      stats="acc=-- ACER=--" if validation else "loss=-- acc=--",
                                      speed="--", peak="--")

    def start(self):
        self.started = time.perf_counter()
        self.progress.start_task(self.task)
        self.progress.refresh()
        return self

    def __call__(self, *, step, loss, examples, accuracy, acer=None, peak=None):
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        if self.validation:
            acer_text = f"{acer:.2%}" if acer is not None else "--"
            stats = f"acc={accuracy:.2%} ACER={acer_text}"
        else:
            stats = f"loss={loss:.4f} acc={accuracy:.2%}"
        self.progress.update(self.task, completed=step, stats=stats,
                             speed=f"{examples / elapsed:.1f}",
                             peak=f"{peak:.2f}" if peak is not None else "--")
