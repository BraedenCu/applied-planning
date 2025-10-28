from __future__ import annotations

from typing import Protocol


class PerceptionAdapter(Protocol):  # pragma: no cover - placeholder interface
    def get_image(self, camera_name: str = "cam0"):
        """Return an RGB image (H, W, 3) as a numpy array."""
        ...
