from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
from src.dynamics_library.system import BaseSystem


@dataclass
class BaseEvent(ABC):
    time: float
    save_state: bool = True
    verbosity: int = 1
    system: Optional[BaseSystem] = None

    def process(self, system: BaseSystem) -> Tuple[BaseEvent, BaseSystem]:
        if self.verbosity > 0:
            print(f"[t={self.time:07.3f}s] {self.get_process_string()}", end="")

        # Advance system to event time
        system = system.predict(self.time)

        # Event-specific behavior
        self.update(system)

        # Save state snapshot if requested
        if self.save_state:
            self.system = system.copy()

        if self.verbosity > 0:
            print(" done")

        return self, system

    @abstractmethod
    def update(self, system: BaseSystem) -> None:
        pass

    @staticmethod
    def get_process_string() -> str:
        return "Processing event:\n"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} at t={self.time:.3f}>"

    def __lt__(self, other: "BaseEvent") -> bool:
        return self.time < other.time


@dataclass
class StateUpdateEvent(BaseEvent):
    def update(self, system: BaseSystem) -> None:

        system.time = self.time

    @staticmethod
    def get_process_string() -> str:
        return "Updating state"


@dataclass
class LogEvent(BaseEvent):
    def update(self, system: BaseSystem) -> None:
        print(f" [LOG] {system.state}", end='')

    @staticmethod
    def get_process_string() -> str:
        return "Logging state"
