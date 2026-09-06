"""Direct-instrumentation counters, shared across every Architecture 4R module.

Why this exists
----------------
The project's own history has a documented case of a compute-saving claim
that was never actually measured -- Architecture 1's efficiency figure was
exit-depth arithmetic, not a count of work performed (`ARCHITECTURES.md`
SS3, "measured rather than asserted"). Architecture 2 and 3 fixed this by
counting real operations; `ppr_push.py`'s `ledger` dict is one such counter,
built ad hoc per module. This module is the single shared counting primitive
so `local_push.py` and the Phase-1 driver can't drift into two different
definitions of "an edge was traversed."
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class OpCounter:
    edges: int = 0
    nodes: int = 0
    queue_pops: int = 0
    madds: int = 0
    phase_s: dict = field(default_factory=dict)
    peak_mem_bytes: int = 0

    def count_edge(self, n: int = 1) -> None:
        self.edges += n

    def count_node_visit(self, n: int = 1) -> None:
        self.nodes += n

    def count_queue_pop(self, n: int = 1) -> None:
        self.queue_pops += n

    def count_madd(self, n: int = 1) -> None:
        self.madds += n

    @contextmanager
    def phase(self, name: str):
        """Wall-clock (and, if torch+CUDA is available, peak-memory) for one phase.

        Accumulates into `phase_s[name]` across repeated calls with the same
        name, so a phase entered multiple times per query (e.g. "diffusion"
        across several checkpoints) reports total time, not just the last call.
        """
        peak_before = self._cuda_peak_bytes()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self.phase_s[name] = self.phase_s.get(name, 0.0) + elapsed
            peak_after = self._cuda_peak_bytes()
            if peak_after is not None:
                self.peak_mem_bytes = max(self.peak_mem_bytes, peak_after)
            del peak_before

    @staticmethod
    def _cuda_peak_bytes():
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated()
        except Exception:
            pass
        return None

    def snapshot(self) -> dict:
        return {
            "edges": self.edges,
            "nodes": self.nodes,
            "queue_pops": self.queue_pops,
            "madds": self.madds,
            "phase_s": dict(self.phase_s),
            "peak_mem_bytes": self.peak_mem_bytes,
        }

    def merge(self, other: "OpCounter") -> None:
        """Fold another counter's totals into this one (e.g. per-query -> per-batch)."""
        self.edges += other.edges
        self.nodes += other.nodes
        self.queue_pops += other.queue_pops
        self.madds += other.madds
        for name, secs in other.phase_s.items():
            self.phase_s[name] = self.phase_s.get(name, 0.0) + secs
        self.peak_mem_bytes = max(self.peak_mem_bytes, other.peak_mem_bytes)
