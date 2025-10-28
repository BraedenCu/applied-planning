from __future__ import annotations

# No-op ROS 2 adapter for Phase 1 (macOS without ROS 2).
# Provides a placeholder API that can be implemented with rclpy later.

from typing import Any, Callable


class Ros2Adapter:  # pragma: no cover - placeholder
    def __init__(self) -> None:
        pass

    def publish(self, topic: str, msg: Any) -> None:
        return None

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        return None
