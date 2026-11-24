from __future__ import annotations

from typing import Any

from .data import make_swift_example


class SAMAMultiViewCollator:
    """Encode clean/aug/noisy/safety views through the native ms-swift Qwen-VL template."""

    def __init__(self, template, cfg: dict):
        self.template = template
        self.cfg = cfg
        self.template.set_mode("train")

    def _encode_batch(self, rows: list[dict], view: str, safety: bool = False):
        encoded = [self.template.encode(make_swift_example(row, view, self.cfg, safety=safety)) for row in rows]
        return self.template.data_collator(encoded)

    def __call__(self, rows: list[dict]) -> dict[str, Any]:
        if not rows:
            raise ValueError("Empty batch")
        route_groups = {str(row["adapter_group"]) for row in rows}
        low_resource_flags = {bool(row["is_low_resource"]) for row in rows}
        if len(route_groups) != 1:
            raise ValueError(f"A batch must use one adapter group, got {route_groups}")
        if len(low_resource_flags) != 1:
            raise ValueError(f"A batch must be homogeneous in is_low_resource, got {low_resource_flags}")

        clean = self._encode_batch(rows, "clean")
        is_low_resource = next(iter(low_resource_flags))
        aug = self._encode_batch(rows, "aug") if is_low_resource else None
        noisy = self._encode_batch(rows, "noisy") if is_low_resource else None

        unsafe_rows = [row for row in rows if bool(row.get("is_unsafe", False))]
        safety = self._encode_batch(unsafe_rows, "clean", safety=True) if unsafe_rows else None

        return {
            "clean": clean,
            "aug": aug,
            "noisy": noisy,
            "safety": safety,
            "meta": {
                "adapter_group": next(iter(route_groups)),
                "is_low_resource": is_low_resource,
                "unsafe_count": len(unsafe_rows),
            },
        }
