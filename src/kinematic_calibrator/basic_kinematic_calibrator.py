from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from beartype import beartype
from lieink.annotations import NDArray_1D, NDArray_N_4_4, RealScalar
from lieink.atoms import SE3
from lieink.containers import Container
from lieink.utils import get_Type
from rich import print
from rich.text import Text


@beartype
class BasicKinematicCalibrator[T](ABC):
    def __init__(self, mechanism: T) -> None:

        self.mechanism_type = get_Type(mechanism)
        self.history_mechanisms = Container[T](self.mechanism_type)
        self.update(mechanism)

    def update(self, mechanism: T) -> None:
        self.mechanism = mechanism
        self.history_mechanisms.append(mechanism)

    @abstractmethod
    def calculate_error(self):
        pass

    @abstractmethod
    def calculate_errorb(self):
        pass

    @abstractmethod
    def calibrate(self):
        pass

    def evaluate(
        self,
        ctrls: Iterable[Iterable[NDArray_1D | Iterable[RealScalar]]],
        actual_poses: NDArray_N_4_4,
    ) -> tuple[NDArray_1D, NDArray_1D]:

        actual_poses = SE3.reshapeb(actual_poses)

        num = len(self.history_mechanisms)
        orient_errors = np.zeros(num)
        position_errors = np.zeros(num)

        for i in range(len(self.history_mechanisms)):
            errors = self.calculate_errorb(ctrls, actual_poses, mechanism_index=i)  # type: ignore
            orient_errors[i] = np.mean(np.linalg.norm(errors[:, :3], axis=1))
            position_errors[i] = np.mean(np.linalg.norm(errors[:, 3:], axis=1))

        return orient_errors, position_errors

    def print(self):
        string = Text()
        string.append(type(self).__name__, style="bold bright_blue").append(
            "[", style="bold"
        ).append(self.mechanism_type.__name__, style="bold bright_magenta").append(
            "]", style="bold"
        )
        if len(self.history_mechanisms) == 1:
            string.append(", which has not been executed yet.")
        else:
            string.append(
                f", which has been executed {len(self.history_mechanisms)} times."
            )
        print(string)
