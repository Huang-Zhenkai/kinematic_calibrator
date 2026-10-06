from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Literal, overload

import lieink
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
    NDArray_N_6_M,
    NDArray_N_M_4_4,
    NDArrayBool_1D,
    RealScalar,
)
from lieink.atoms import SE3, Ad, Twist
from lieink.containers import Container
from lieink.mechanisms.localpoe.parallel_mechanisms import Limb, ParallelMechanism
from lieink.utils import NDArray_3Dto2D, get_Type

from kinematic_calibrator.basic_kinematic_calibrator import BasicKinematicCalibrator
from kinematic_calibrator.for_mechanism_localpoe.for_serial_mechanism import (
    KinematicCalibrator as SerialKinematicCalibrator,
)


@beartype
class LimbKinematicCalibrator[T: Limb](SerialKinematicCalibrator[T]):
    def _calculate_gamma(
        self, localposes: NDArray_N_M_4_4, null_spaces_of_Jump: NDArray_N_6_M
    ) -> NDArray_2D:

        null_space_dim = null_spaces_of_Jump.shape[-1]
        data_num = null_spaces_of_Jump.shape[0]
        localposes_Ad = SE3.toAdcb(localposes.reshape(-1, 4, 4)).reshape(
            data_num, -1, 6, 6
        )

        gamma_ = np.einsum("nji,nmjk->nmik", null_spaces_of_Jump, localposes_Ad)
        gamma_ = gamma_.reshape(-1, null_space_dim, 6)
        gamma_ = NDArray_3Dto2D(gamma_, shape=(-1, self.mechanism.joint_num + 1))

        return gamma_

    def __calculate_redundant_block_of_2ump(
        self, begin_index: int, now_index: int
    ) -> NDArray_2D:
        if begin_index < 0:
            redundant_parameter = np.empty((6, 0))
        else:
            if begin_index == now_index:
                Adg_k = self.mechanism.kinematic_parameters_Ad[now_index + 1]
                next_joint_twist = Adg_k[:, -1:]
                now_joint_twist = np.array([[0], [0], [0], [0], [0], [1]])

                redundant_parameter = (
                    Twist.toadc(lieink.utils.DELTA @ now_joint_twist)
                    @ lieink.utils.DELTA
                    @ next_joint_twist
                )

            elif begin_index == now_index - 1:
                Adg_k = self.mechanism.kinematic_parameters_Ad[now_index]
                last_joint_twist = Ad.invc(Adg_k)[:, -1:]
                now_joint_twist = np.array([[0], [0], [0], [0], [0], [1]])

                redundant_parameter = Twist.toadc(
                    lieink.utils.DELTA @ last_joint_twist
                ) @ (lieink.utils.DELTA @ now_joint_twist)

            else:
                redundant_parameter = np.empty((6, 0))

        return redundant_parameter

    def __calculate_redundant_block_of_2ump_1mp(
        self, begin_index: int, measurable_index: int, now_index: int
    ) -> NDArray_2D:
        if begin_index < 0:
            redundant_parameter = np.empty((6, 0))
        else:
            if measurable_index == 0:
                if begin_index == now_index:
                    Adg_k = self.mechanism.kinematic_parameters_Ad[now_index + 1]
                    next_joint_twist = Adg_k[:, -1:]
                    Adg_kplus1 = self.mechanism.kinematic_parameters_Ad[now_index + 2]
                    the_joint_twist_after_next = (Adg_k @ Adg_kplus1)[:, -1:]

                    redundant_parameter = Twist.toadc(
                        lieink.utils.DELTA @ next_joint_twist
                    ) @ (lieink.utils.DELTA @ the_joint_twist_after_next)

                else:
                    redundant_parameter = self.__calculate_redundant_block_of_2ump(
                        begin_index + 1, now_index
                    )

            elif measurable_index == 1:
                if begin_index == now_index:
                    Adg_k = self.mechanism.kinematic_parameters_Ad[now_index + 1]
                    Adg_kplus1 = self.mechanism.kinematic_parameters_Ad[now_index + 2]
                    the_joint_twist_after_next = (Adg_k @ Adg_kplus1)[:, -1:]
                    now_joint_twist = np.array([[0], [0], [0], [0], [0], [1]])

                    redundant_parameter = Twist.toadc(
                        lieink.utils.DELTA @ now_joint_twist
                    ) @ (lieink.utils.DELTA @ the_joint_twist_after_next)

                elif begin_index == now_index - 1:
                    Adg_k = self.mechanism.kinematic_parameters_Ad[now_index]
                    Adg_kplus1 = self.mechanism.kinematic_parameters_Ad[now_index + 1]
                    last_joint_twist = Ad.invc(Adg_k)[:, -1:]
                    next_joint_twist = Adg_kplus1[:, -1:]

                    redundant_parameter = Twist.toadc(
                        lieink.utils.DELTA @ last_joint_twist
                    ) @ (lieink.utils.DELTA @ next_joint_twist)

                elif begin_index == now_index - 2:
                    Adg_k = self.mechanism.kinematic_parameters_Ad[now_index - 1]
                    Adg_kplus1 = self.mechanism.kinematic_parameters_Ad[now_index]
                    the_joint_twist_before_last = Ad.invc(Adg_k @ Adg_kplus1)[:, -1:]
                    now_joint_twist = np.array([[0], [0], [0], [0], [0], [1]])

                    redundant_parameter = Twist.toadc(
                        lieink.utils.DELTA @ the_joint_twist_before_last
                    ) @ (lieink.utils.DELTA @ now_joint_twist)

                else:
                    redundant_parameter = np.empty((6, 0))
            else:
                if begin_index == now_index or begin_index == now_index - 1:
                    redundant_parameter = self.__calculate_redundant_block_of_2ump(
                        begin_index, now_index
                    )
                elif begin_index == now_index - 2:
                    Adg_k = self.mechanism.kinematic_parameters_Ad[now_index - 1]
                    Adg_kplus1 = self.mechanism.kinematic_parameters_Ad[now_index]
                    the_last_joint_twist = Ad.invc(Adg_kplus1)[:, -1:]
                    the_joint_twist_before_last = Ad.invc(Adg_k @ Adg_kplus1)[:, -1:]

                    redundant_parameter = Twist.toadc(
                        lieink.utils.DELTA @ the_joint_twist_before_last
                    ) @ (lieink.utils.DELTA @ the_last_joint_twist)
                else:
                    redundant_parameter = np.empty((6, 0))

        return redundant_parameter

    def _calculate_separation_operator(self) -> NDArray_2D:

        jn = self.mechanism.joint_num
        mask_unm = self.mechanism.mask_unmeasurable_joints
        pjoint_num = self.mechanism.joint_types.count("P")
        jts = self.mechanism.joint_types
        delta = self._calculate_delta()

        rps_from_unmeasurable_joints = (
            np.linalg.inv(delta)
            @ (
                slg.block_diag(np.zeros((6, 1)), *self.mechanism.joint_twists)[:, 1:][
                    :, mask_unm
                ]
            )
        )

        redundant_block_from_topology = []
        case1_begin_index = -1
        case2_begin_index = -1
        case2_measurable = -1
        case3 = False

        if pjoint_num == 3:
            for i in range(jn - 2):
                adjacent_joints = jts[i : i + 3]
                if adjacent_joints.count("P") == 3:
                    unmeasured_num = np.sum(mask_unm[i : i + 3])
                    if unmeasured_num == 3:
                        case3 = True
                    elif unmeasured_num == 2:
                        case2_begin_index = i
                        case2_measurable = int(np.argmin(mask_unm[i : i + 3]))
                    else:
                        pass
                    break
        if pjoint_num >= 2 and case2_begin_index == -1 and case3 is False:
            for i in range(jn - 1):
                adjacent_joints = jts[i : i + 2]
                if adjacent_joints.count("P") == 2 and np.sum(mask_unm[i : i + 2]) == 2:
                    case1_begin_index = i
                    break

        redundant_block_for_r = (
            np.eye(6)[:, [2, 3, 4, 5]] if case3 else np.eye(6)[:, [2, 5]]
        )
        redundant_block_for_p = np.eye(6) if case3 else np.eye(6)[:, [2, 3, 4, 5]]

        # from serial structure
        for i in range(jn):
            if self.mechanism.joint_types[i] == "R":
                redundant_block_i = redundant_block_for_r
            else:
                redundant_block_i = redundant_block_for_p
                if case1_begin_index >= 0:
                    redundant_parameter_from_topology = (
                        self.__calculate_redundant_block_of_2ump(case1_begin_index, i)
                    )
                    redundant_block_i = np.hstack(
                        (
                            redundant_block_i,
                            redundant_parameter_from_topology,
                        )
                    )
                elif case2_begin_index >= 0:
                    redundant_parameter_from_topology = (
                        self.__calculate_redundant_block_of_2ump_1mp(
                            case2_begin_index, case2_measurable, i
                        )
                    )
                    redundant_block_i = np.hstack(
                        (
                            redundant_block_i,
                            redundant_parameter_from_topology,
                        )
                    )
                else:
                    pass
            redundant_block_from_topology.append(redundant_block_i)
        rps_from_topology = slg.block_diag(
            *redundant_block_from_topology,
            np.zeros((6, 1)),
        )[:, :-1]

        rps_from_kinematic_configuration = []
        qs = np.random.uniform(-np.pi, np.pi, (42, jn))
        _, vjacobians, local_poses = self.mechanism.forward_kinematicsb(
            qs, return_vjacobian=True, return_local_poses=True
        )
        twists_ump = vjacobians[:, mask_unm]
        null_spaces_of_Jump = slg.null_space(twists_ump.squeeze(-1))
        gamma_ = self._calculate_gamma(local_poses, null_spaces_of_Jump)

        for i in range(jn):
            gamma_i = gamma_[:, 6 * i + 6 : 6 * i + 12].reshape(42, -1, 6)
            Adexp_theta_xi = Twist.expcb(
                self.mechanism.joint_twists[i].repeat(42, axis=1), -qs[:, i], expto="Ad"
            )
            A_i = NDArray_3Dto2D(gamma_i @ (np.eye(6) - Adexp_theta_xi), shape=(-1, 1))
            _, s, vt = np.linalg.svd(A_i)
            rank = int(np.sum(s > 1e-12))
            x_i = vt[rank:].conj().T
            rbt_i = redundant_block_from_topology[i]
            if x_i.shape[1] > rbt_i.shape[1]:
                redundant_block_from_kinematic_configuration = slg.orth(
                    x_i - rbt_i @ slg.pinv(rbt_i) @ x_i
                )
                rps_i = np.zeros(
                    (
                        rps_from_topology.shape[0],
                        redundant_block_from_kinematic_configuration.shape[1],
                    )
                )
                rps_i[6 * i : 6 * (i + 1)] = (
                    redundant_block_from_kinematic_configuration
                )
                rps_from_kinematic_configuration.append(rps_i)
        if len(rps_from_kinematic_configuration) > 0:
            rps_from_kinematic_configuration = np.hstack(
                rps_from_kinematic_configuration
            )
        else:
            rps_from_kinematic_configuration = np.empty(
                (
                    rps_from_topology.shape[0],
                    0,
                )
            )

        rps = np.hstack(
            (
                rps_from_unmeasurable_joints,
                rps_from_topology,
                rps_from_kinematic_configuration,
            )
        )

        self.redundant_parameters_from_unmeasurable_joints = (
            rps_from_unmeasurable_joints
        )
        self.redundant_parameters_from_topology = rps_from_topology
        self.redundant_parameters_from_kinematic_configuration = (
            rps_from_kinematic_configuration
        )
        self.redundant_parameters = rps

        sop = slg.null_space(rps.T)

        return sop

    @overload
    def calculate_error(
        self,
        ctrl_ideal: NDArray_1D | Iterable[RealScalar],
        actual_pose: SE3 | NDArray_4_4,
        ctrl_actuated_joints: None | NDArray_1D | Iterable[RealScalar] = None,
        return_NDArray: Literal[False] = False,
        mechanism_index: int = -1,
    ) -> Twist: ...

    @overload
    def calculate_error(
        self,
        ctrl_ideal: NDArray_1D | Iterable[RealScalar],
        actual_pose: SE3 | NDArray_4_4,
        ctrl_actuated_joints: None | NDArray_1D | Iterable[RealScalar] = None,
        return_NDArray: Literal[True] = True,
        mechanism_index: int = -1,
    ) -> NDArray_6_1: ...

    def calculate_error(
        self,
        ctrl_ideal: NDArray_1D | Iterable[RealScalar],
        actual_pose: SE3 | NDArray_4_4,
        ctrl_actuated_joints: None | NDArray_1D | Iterable[RealScalar] = None,
        return_NDArray: bool = False,
        mechanism_index: int = -1,
    ) -> Twist | NDArray_6_1:

        ctrl_ideal = np.asarray(ctrl_ideal)
        mechanism = self.history_mechanisms[mechanism_index]
        if isinstance(actual_pose, SE3):
            actual_pose = actual_pose.v
        else:
            actual_pose = SE3._check_shape_and_value(actual_pose)

        target_pose, vjacobian = mechanism.forward_kinematics(
            ctrl_ideal, return_NDArray=True, return_vjacobian=True
        )
        error = SE3.logc(
            actual_pose @ SE3.invc(target_pose), logto="Twist", skip_check=True
        )
        if ctrl_actuated_joints is not None:
            ctrl_actuated_joints = (
                np.asarray(ctrl_actuated_joints)
                - ctrl_ideal[mechanism.mask_measurable_joints]
            )
            error_of_mesurable_joints = (
                vjacobian[mechanism.mask_measurable_joints]
                * ctrl_actuated_joints.reshape(-1, 1, 1)
            ).sum(axis=0)
            error -= error_of_mesurable_joints

        null_space_of_Jump = slg.null_space(
            vjacobian[mechanism.mask_unmeasurable_joints].squeeze(-1)
        )
        error = null_space_of_Jump @ (null_space_of_Jump.T @ error)

        if return_NDArray:
            return error

        return Twist(error)

    def calculate_errorb(
        self,
        ctrls_ideal: NDArray_2D | Iterable[NDArray_1D | Iterable[RealScalar]],
        actual_poses: NDArray_N_4_4,
        ctrls_actuated_joints: None
        | NDArray_2D
        | Iterable[NDArray_1D | Iterable[RealScalar]] = None,
        mechanism_index: int = -1,
    ) -> NDArray_N_6_1:

        ctrls_ideal = np.asarray(ctrls_ideal)
        data_num = ctrls_ideal.shape[0]
        mechanism = self.history_mechanisms[mechanism_index]

        target_poses, vjacobians = mechanism.forward_kinematicsb(
            ctrls_ideal, return_vjacobian=True
        )
        errors = SE3.logcb(
            actual_poses @ SE3.invcb(target_poses), logto="Twist", skip_check=True
        )
        if ctrls_actuated_joints is not None:
            ctrls_actuated_joints = (
                np.asarray(ctrls_actuated_joints)
                - ctrls_ideal[:, mechanism.mask_measurable_joints]
            )
            error_of_mesurable_joints = (
                vjacobians[:, mechanism.mask_measurable_joints]
                * ctrls_actuated_joints.reshape(data_num, -1, 1, 1)
            ).sum(axis=1)
            errors -= error_of_mesurable_joints

        null_spaces_of_Jump = slg.null_space(
            vjacobians[:, mechanism.mask_unmeasurable_joints].squeeze(-1)
        )
        errors = null_spaces_of_Jump @ (null_spaces_of_Jump.swapaxes(1, 2) @ errors)

        return errors

    def calibrate(
        self,
        ctrls_ideal: NDArray_2D | Iterable[NDArray_1D | Iterable[RealScalar]],
        actual_poses: NDArray_N_4_4,
        solver: Callable,
        ctrls_actuated_joints: None
        | NDArray_2D
        | Iterable[NDArray_1D | Iterable[RealScalar]] = None,
    ) -> None:

        ctrls_ideal = np.asarray(ctrls_ideal)
        data_num = ctrls_ideal.shape[0]
        actual_poses = SE3.reshapeb(actual_poses)

        target_poses, vjacobians, localposes = self.mechanism.forward_kinematicsb(
            ctrls_ideal, return_local_poses=True, return_vjacobian=True
        )
        errors = SE3.logcb(
            actual_poses @ SE3.invcb(target_poses), logto="Twist", skip_check=True
        )
        if ctrls_actuated_joints is not None:
            ctrls_actuated_joints = (
                np.asarray(ctrls_actuated_joints)
                - ctrls_ideal[:, self.mechanism.mask_measurable_joints]
            )
            error_of_mesurable_joints = (
                vjacobians[:, self.mechanism.mask_measurable_joints]
                * ctrls_actuated_joints.reshape(data_num, -1, 1, 1)
            ).sum(axis=1)
            errors -= error_of_mesurable_joints

        null_spaces_of_Jump = slg.null_space(
            vjacobians[:, self.mechanism.mask_unmeasurable_joints].squeeze(-1)
        )

        errors = null_spaces_of_Jump.swapaxes(1, 2) @ errors
        errors = NDArray_3Dto2D(errors, shape=(-1, 1))
        gamma = self._calculate_gamma(localposes, null_spaces_of_Jump)

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

        limb = self.mechanism_type(
            self.mechanism.kinematic_parameters
            + Twist.reshapeb(delta_kinematic_parameters),
            self.mechanism.joint_types,
            self.mechanism.joint_actuation_types,
        )

        self.update(limb)


@beartype
class KinematicCalibrator[T: ParallelMechanism](BasicKinematicCalibrator):
    def __init__(self, mechanism: T) -> None:

        self.mechanism_type = get_Type(mechanism)
        self.history_mechanisms = Container[T](self.mechanism_type)
        self.limb_kinematic_calibrators = Container[LimbKinematicCalibrator](
            LimbKinematicCalibrator
        )
        for limb in mechanism.limbs:
            self.limb_kinematic_calibrators.append(LimbKinematicCalibrator(limb))
        self.update(mechanism)

    @overload
    def calculate_error(
        self,
        ctrl: Iterable[NDArray_1D | Iterable[RealScalar]],
        actual_pose: SE3 | NDArray_4_4,
        return_NDArray: Literal[False] = False,
        mechanism_index: int = ...,
        dont_raise: Literal[False] = False,
    ) -> Twist: ...

    @overload
    def calculate_error(
        self,
        ctrl: Iterable[NDArray_1D | Iterable[RealScalar]],
        actual_pose: SE3 | NDArray_4_4,
        return_NDArray: Literal[False] = False,
        mechanism_index: int = ...,
        dont_raise: Literal[True] = True,
    ) -> tuple[Twist, bool]: ...

    @overload
    def calculate_error(
        self,
        ctrl: Iterable[NDArray_1D | Iterable[RealScalar]],
        actual_pose: SE3 | NDArray_4_4,
        return_NDArray: Literal[True] = True,
        mechanism_index: int = ...,
        dont_raise: Literal[False] = False,
    ) -> NDArray_6_1: ...

    @overload
    def calculate_error(
        self,
        ctrl: Iterable[NDArray_1D | Iterable[RealScalar]],
        actual_pose: SE3 | NDArray_4_4,
        return_NDArray: Literal[True] = True,
        mechanism_index: int = ...,
        dont_raise: Literal[True] = True,
    ) -> tuple[NDArray_6_1, bool]: ...

    def calculate_error(
        self,
        ctrl: Iterable[NDArray_1D | Iterable[RealScalar]],
        actual_pose: SE3 | NDArray_4_4,
        return_NDArray: bool = False,
        mechanism_index: int = -1,
        dont_raise: bool = False,
    ) -> Twist | NDArray_6_1 | tuple[Twist, bool] | tuple[NDArray_6_1, bool]:

        mechanism = self.history_mechanisms[mechanism_index]
        if isinstance(actual_pose, SE3):
            actual_pose = actual_pose.v
        else:
            actual_pose = SE3._check_shape_and_value(actual_pose)
        if dont_raise:
            target_pose, success = mechanism.coordinate_pose_by_ctrl(
                ctrl, return_NDArray=True, dont_raise=True
            )
        else:
            target_pose = mechanism.coordinate_pose_by_ctrl(ctrl, return_NDArray=True)
        error = SE3.logc(
            actual_pose @ SE3.invc(target_pose), logto="Twist", skip_check=True
        )

        if return_NDArray:
            if dont_raise:
                return error, success
            return error
        if dont_raise:
            return Twist(error), success
        return Twist(error)

    @overload
    def calculate_errorb(
        self,
        ctrls: Iterable[Iterable[NDArray_1D | Iterable[RealScalar]]],
        actual_poses: NDArray_N_4_4,
        mechanism_index: int = ...,
        dont_raise: Literal[False] = False,
    ) -> NDArray_N_6_1: ...

    @overload
    def calculate_errorb(
        self,
        ctrls: Iterable[Iterable[NDArray_1D | Iterable[RealScalar]]],
        actual_poses: NDArray_N_4_4,
        mechanism_index: int = ...,
        dont_raise: Literal[True] = True,
    ) -> tuple[NDArray_N_6_1, NDArrayBool_1D]: ...

    def calculate_errorb(
        self,
        ctrls: Iterable[Iterable[NDArray_1D | Iterable[RealScalar]]],
        actual_poses: NDArray_N_4_4,
        mechanism_index: int = -1,
        dont_raise: bool = False,
    ) -> NDArray_N_6_1 | tuple[NDArray_N_6_1, NDArrayBool_1D]:
        mechanism = self.history_mechanisms[mechanism_index]
        actual_poses = SE3.reshapeb(actual_poses)
        if dont_raise:
            target_poses, success = mechanism.coordinate_pose_by_ctrlb(
                ctrls, dont_raise=True
            )
            errors = SE3.logcb(
                actual_poses @ SE3.invcb(target_poses), logto="Twist", skip_check=True
            )
            return errors, success
        else:
            target_poses = mechanism.coordinate_pose_by_ctrlb(
                ctrls, dont_raise=dont_raise
            )
            errors = SE3.logcb(
                actual_poses @ SE3.invcb(target_poses), logto="Twist", skip_check=True
            )
            return errors

    def calibrate(
        self,
        ctrls_ideal: Iterable[Iterable[NDArray_1D | Iterable[RealScalar]]],
        actual_poses: NDArray_N_4_4,
        solver: Callable | Iterable[Callable],
        ctrls_actuated_joints: None
        | Iterable[Iterable[NDArray_1D | Iterable[RealScalar]]] = None,
    ):
        mechanism = self.mechanism
        _, ctrls = mechanism.coordinate_pose_by_ctrlb(ctrls_ideal, return_ctrls=True)
        mechanism_ = deepcopy(mechanism)

        if isinstance(solver, Iterable):
            solver_ = list(solver)
        else:
            solver_ = [solver] * self.mechanism.limb_num

        for i in range(len(self.limb_kinematic_calibrators)):
            ctrls_i = [ctrls[j][i] for j in range(len(ctrls))]
            if ctrls_actuated_joints is not None:
                ctrls_actuated_joints_i = [
                    ctrls_actuated_joints[j][i]  # type: ignore
                    for j in range(len(ctrls_actuated_joints))  # type: ignore
                ]
            else:
                ctrls_actuated_joints_i = None
            self.limb_kinematic_calibrators[i].calibrate(
                ctrls_i, actual_poses, solver_[i], ctrls_actuated_joints_i
            )
            mechanism_._limbs[i] = self.limb_kinematic_calibrators[i].mechanism

        self.update(mechanism_)
