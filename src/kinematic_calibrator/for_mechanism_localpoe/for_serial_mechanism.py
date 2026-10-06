from collections.abc import Callable, Iterable
from typing import Literal, overload

import numpy as np
import scipy.linalg as slg
from beartype import beartype
from lieink.annotations import (
    NDArray_1D,
    NDArray_2D,
    NDArray_4_4,
    NDArray_6_1,
    NDArray_N_4_4,
    NDArray_N_6_1,
    NDArray_N_M_4_4,
    RealScalar,
)
from lieink.atoms import SE3, Twist
from lieink.mechanisms.localpoe.serial_mechanisms import SerialMechanism
from lieink.utils import NDArray_3Dto2D

from kinematic_calibrator.basic_kinematic_calibrator import (
    BasicKinematicCalibrator,
)


@beartype
class KinematicCalibrator[T: SerialMechanism](BasicKinematicCalibrator[T]):
    def update(self, mechanism: T) -> None:
        super().update(mechanism)
        self.delta = self._calculate_delta()
        self.separation_operator = self._calculate_separation_operator()
        self.left_jacobian = self._calculate_left_jacobian()

    def _calculate_gamma(self, localposes: NDArray_N_M_4_4) -> NDArray_2D:
        localposes_Ad = SE3.toAdcb(localposes.reshape(-1, 4, 4))
        gamma = NDArray_3Dto2D(localposes_Ad, shape=(-1, self.mechanism.joint_num + 1))
        return gamma

    def _calculate_delta(self) -> NDArray_2D:

        serialmechanism = self.mechanism
        delta = slg.block_diag(*serialmechanism.kinematic_parameters_Ad)
        for i in range(serialmechanism.joint_num):
            delta[6 * i + 6 : 6 * i + 12, 6 * i : 6 * i + 6] = -np.eye(6)

        return delta

    def _calculate_separation_operator(self) -> NDArray_2D:

        serialmechanism = self.mechanism
        separation_operator = []
        redundant_parameters = []
        for i in range(serialmechanism.joint_num):
            if serialmechanism.joint_types[i] == "R":
                separation_operator.append(np.eye(6)[:, [0, 1, 3, 4]])
                redundant_parameters.append(np.eye(6)[:, [2, 5]])
            else:
                separation_operator.append(np.eye(6)[:, [0, 1]])
                redundant_parameters.append(np.eye(6)[:, [2, 3, 4, 5]])

        separation_operator = slg.block_diag(*separation_operator, np.eye(6))
        self.redundant_parameters = slg.block_diag(
            *redundant_parameters, np.zeros((6, 1))
        )[:, :-1]

        return separation_operator

    def _calculate_left_jacobian(self) -> NDArray_2D:
        return slg.block_diag(
            *Twist.left_jacobiancb(self.mechanism.kinematic_parameters)
        )

    @overload
    def calculate_error(
        self,
        ctrl: NDArray_1D | Iterable[RealScalar],
        actual_pose: SE3 | NDArray_4_4,
        return_NDArray: Literal[False] = False,
        mechanism_index: int = ...,
    ) -> Twist: ...

    @overload
    def calculate_error(
        self,
        ctrl: NDArray_1D | Iterable[RealScalar],
        actual_pose: SE3 | NDArray_4_4,
        return_NDArray: Literal[True],
        mechanism_index: int = ...,
    ) -> NDArray_6_1: ...

    def calculate_error(
        self,
        ctrl: NDArray_1D | Iterable[RealScalar],
        actual_pose: SE3 | NDArray_4_4,
        return_NDArray: bool = False,
        mechanism_index: int = -1,
    ) -> Twist | NDArray_6_1:

        ctrl = np.asarray(ctrl)
        mechanism = self.history_mechanisms[mechanism_index]
        if isinstance(actual_pose, SE3):
            actual_pose = actual_pose.v
        else:
            actual_pose = SE3._check_shape_and_value(actual_pose)
        target_pose = mechanism.forward_kinematics(ctrl, return_NDArray=True)
        error = SE3.logc(
            actual_pose @ SE3.invc(target_pose), logto="Twist", skip_check=True
        )

        if return_NDArray:
            return error

        return Twist(error)

    def calculate_errorb(
        self,
        ctrls: NDArray_2D | Iterable[NDArray_1D | Iterable[RealScalar]],
        actual_poses: NDArray_N_4_4,
        mechanism_index: int = -1,
    ) -> NDArray_N_6_1:

        actual_poses = SE3.reshapeb(actual_poses)
        mechanism = self.history_mechanisms[mechanism_index]
        target_poses = mechanism.forward_kinematicsb(ctrls)
        errors = SE3.logcb(
            actual_poses @ SE3.invcb(target_poses), logto="Twist", skip_check=True
        )

        return errors

    def calibrate(
        self,
        ctrls: NDArray_2D | Iterable[NDArray_1D | Iterable[RealScalar]],
        actual_poses: NDArray_N_4_4,
        solver: Callable,
    ) -> None:

        ctrls = np.asarray(ctrls)
        actual_poses = SE3.reshapeb(actual_poses)

        target_poses, localposes = self.mechanism.forward_kinematicsb(
            ctrls, return_local_poses=True
        )
        errors = SE3.logcb(
            actual_poses @ SE3.invcb(target_poses), logto="Twist", skip_check=True
        )
        errors = NDArray_3Dto2D(errors, shape=(-1, 1))
        gamma = self._calculate_gamma(localposes)

        delta_p = solver(gamma @ self.delta @ self.separation_operator, errors)
        delta_kinematic_parameters = (
            (
                np.linalg.inv(self.left_jacobian)
                @ self.delta
                @ self.separation_operator
                @ delta_p
            )
            .reshape(-1, 6)
            .T
        )
        mechanism = self.mechanism_type(
            self.mechanism.kinematic_parameters
            + Twist.reshapeb(delta_kinematic_parameters),
            self.mechanism.joint_types,
        )

        self.update(mechanism)
