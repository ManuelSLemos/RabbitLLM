import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# Mapping of raw metric keys → human-readable category labels.
# Metrics not listed here are grouped under "Other".
_CATEGORIES: Dict[str, str] = {
    "load_safe_tensor": "Disk I/O",
    "load_safe_tensor_gds": "Disk I/O (GDS)",
    "load_safe_tensor_cpu_wait": "CPU load wait",
    "pin_memory_to_trigger_load": "Pin memory",
    "create_layer_from_state_dict": "CPU->GPU transfer",
    "create_layer_from_safe_tensor": "CPU->GPU transfer",
    "kick_off_load_cpu": "Pipeline overhead",
    "compression_time": "Decompression",
    "forward_per_layer": "GPU forward compute",
    # small-layer offload paths
    "small_layer_cache_populate": "Small layer cache populate",
    "small_layer_cache_hit_clone": "Small layer cache hit (clone)",
    # post-embed RoPE computation
    "position_embeddings_compute": "RoPE position embeddings",
    # tied lm_head weight assignment
    "tied_lm_head_load": "Tied lm_head load",
    # async transfer stream stall during Phase B (decode bottleneck)
    "transfer_stream_sync_wait": "Transfer sync stall",
}

# Display order for the summary table.
_CATEGORY_ORDER: List[str] = [
    "Disk I/O",
    "Disk I/O (GDS)",
    "CPU load wait",
    "Pin memory",
    "CPU->GPU transfer",
    "Decompression",
    "GPU forward compute",
    "Small layer cache populate",
    "Small layer cache hit (clone)",
    "RoPE position embeddings",
    "Tied lm_head load",
    "Transfer sync stall",
    "Pipeline overhead",
    "Other",
]


class LayeredProfiler:
    def __init__(self, print_memory: bool = False):
        self.profiling_time_dict: Dict[str, List[float]] = {}
        self.print_memory = print_memory
        self.min_free_mem = 1024 * 1024 * 1024 * 1024

    def add_profiling_time(self, item: str, time: float) -> None:
        if item not in self.profiling_time_dict:
            self.profiling_time_dict[item] = []
        self.profiling_time_dict[item].append(time)

        if self.print_memory:
            free_mem = torch.cuda.mem_get_info()[0]
            self.min_free_mem = min(self.min_free_mem, free_mem)
            logger.debug(
                "free vmem @%s: %.02fGB, min free: %.02fGB",
                item,
                free_mem / 1024 / 1024 / 1024,
                self.min_free_mem / 1024 / 1024 / 1024,
            )

    def clear_profiling_time(self) -> None:
        for item in self.profiling_time_dict:
            self.profiling_time_dict[item] = []

    def print_profiling_time(self) -> None:
        """Legacy: log raw totals per metric key."""
        for item, times in self.profiling_time_dict.items():
            logger.info("total time for %s: %s", item, sum(times))

    def report(self, label: str = "") -> str:
        """Return a formatted profiling report as a string and log it.

        Groups raw metric keys into categories, computes totals and percentages,
        and identifies the bottleneck (the single category consuming the most time).

        Args:
            label: Optional tag shown in the header (e.g. "prefill" or "decode t=3").

        Returns:
            The report as a plain string (also logged at INFO level).
        """
        # --- Aggregate raw metrics into categories ---
        category_totals: Dict[str, float] = {}
        category_counts: Dict[str, int] = {}

        for key, times in self.profiling_time_dict.items():
            if not times:
                continue
            cat = _CATEGORIES.get(key, "Other")
            category_totals[cat] = category_totals.get(cat, 0.0) + sum(times)
            category_counts[cat] = category_counts.get(cat, 0) + len(times)

        if not category_totals:
            return "(no profiling data)"

        grand_total = sum(category_totals.values())
        if grand_total == 0:
            return "(all zeros)"

        bottleneck = max(category_totals, key=lambda c: category_totals[c])

        # --- Build table rows in display order ---
        rows = []
        for cat in _CATEGORY_ORDER:
            if cat not in category_totals:
                continue
            total = category_totals[cat]
            count = category_counts[cat]
            avg_ms = (total / count * 1000) if count > 0 else 0.0
            pct = total / grand_total * 100
            marker = " <-- BOTTLENECK" if cat == bottleneck else ""
            rows.append((cat, total, count, avg_ms, pct, marker))

        # Column widths
        cat_w = max(len(r[0]) for r in rows) + 2
        header = label or "forward pass"

        lines = [
            f"  Profiler report [{header}]  (total={grand_total:.3f}s)",
            "  " + "-" * (cat_w + 46),
            f"  {'Category':<{cat_w}}  {'Total':>8}  {'Layers':>6}  {'Avg/layer':>10}  {'%':>5}",
            "  " + "-" * (cat_w + 46),
        ]
        for cat, total, count, avg_ms, pct, marker in rows:
            lines.append(
                f"  {cat:<{cat_w}}  {total:>7.3f}s  {count:>6}  {avg_ms:>8.1f}ms  {pct:>4.1f}%{marker}"
            )
        lines.append("  " + "-" * (cat_w + 46))
        lines.append(
            f"  Bottleneck: {bottleneck} ({category_totals[bottleneck]:.3f}s,"
            f" {category_totals[bottleneck] / grand_total * 100:.0f}% of layer-loop time)"
        )

        report_str = "\n".join(lines)
        logger.info("\n%s", report_str)
        return report_str

    def accumulate_into(self, target: "LayeredProfiler") -> None:
        """Merge this profiler's data into ``target`` (for cross-step aggregation)."""
        for key, times in self.profiling_time_dict.items():
            if key not in target.profiling_time_dict:
                target.profiling_time_dict[key] = []
            target.profiling_time_dict[key].extend(times)
