from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
from src.dynamics_library.system import BaseSystem


@dataclass
class BaseEvent(ABC):
    """
    Base class for time-ordered events in a simulation.

    Attributes:
        time (float): 
            The time at which the event occurs.
        save_state (bool, optional): 
            Whether to save the system state after the event. Defaults to True.
        verbosity (int, optional): 
            Controls logging output. 0 = silent, 1 = show basic progress. Defaults to 1.
        system (Optional[BaseSystem], optional): 
            A snapshot of the system after processing this event. Only saved if `save_state` is True.
    """
    time: float
    save_state: bool = True
    verbosity: int = 1
    system: Optional[BaseSystem] = None

    def process(self, system: BaseSystem) -> Tuple[BaseEvent, BaseSystem]:
        """
        Process the event by advancing the system to the event time, applying the event-specific update,
        and optionally saving the resulting system state.

        Args:
            system (BaseSystem): The current system state.

        Returns:
            Tuple[BaseEvent, BaseSystem]: The updated event and system.
        """
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
        """
        Abstract method that defines event-specific behavior.

        Args:
            system (BaseSystem): The system to update.
        """
        pass

    @staticmethod
    def get_process_string() -> str:
        """
        Returns a string used for logging during processing.

        Returns:
            str: The logging message.
        """
        return "Processing event:\n"

    def __repr__(self) -> str:
        """
        String representation of the event.

        Returns:
            str: A compact string showing the event type and time.
        """
        return f"<{self.__class__.__name__} at t={self.time:.3f}>"

    def __lt__(self, other: "BaseEvent") -> bool:
        """
        Compare events by time for sorting.

        Args:
            other (BaseEvent): Another event.

        Returns:
            bool: True if this event occurs before the other.
        """
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
