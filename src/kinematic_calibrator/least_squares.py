import numpy as np
from lieink.annotations import NDArray_1D, NDArray_2D


def ordinary_least_squares(A: NDArray_2D, b: NDArray_1D | NDArray_2D) -> NDArray_2D:
    return np.linalg.lstsq(A, b)[0]


def total_least_squares(A: NDArray_2D, b: NDArray_1D | NDArray_2D) -> NDArray_2D:

    B = np.hstack((-b.reshape(-1, 1), A))
    _, _, Vt = np.linalg.svd(B)
    V = Vt.T
    p = B.shape[1] - 1

    Vp = V[:, p:]
    v1 = Vp[:1].T
    Vp_ = Vp[1:]

    return (Vp_ @ v1) / np.sum(v1**2)


def truncated_total_least_squares(
    A: NDArray_2D, b: NDArray_1D | NDArray_2D, p: int | None = None
) -> NDArray_2D:

    B = np.hstack((-b.reshape(-1, 1), A))
    _, _, Vt = np.linalg.svd(B)
    V = Vt.T
    if p is None:
        p = B.shape[1] - 1
    elif p > B.shape[1] - 1:
        raise ValueError("p must be less than or equal to A.shape[1]-1")
    else:
        pass

    Vp = V[:, p:]
    v1 = Vp[:1].T
    Vp_ = Vp[1:]

    return (Vp_ @ v1) / np.sum(v1**2)


def _householder_vec(x):
    """
    计算并返回 Householder 反射的归一化向量 v。
    不显式构造 H 矩阵，极大减少浮点误差。
    """
    x = np.asarray(x, dtype=float)
    norm_x = np.linalg.norm(x)
    if norm_x == 0:
        return np.zeros_like(x)

    v = x.copy()
    # 符号选择保证数值稳定性，避免相消误差 (Cancellation error)
    sign = 1.0 if v[0] >= 0 else -1.0
    v[0] += sign * norm_x

    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return np.zeros_like(x)

    return v / norm_v


def _transform_to_core_problem(B, c):
    """
    返回 P, 双对角矩阵, 和 Q
    满足 P @ [c_tilde | B_tilde] @ block_diag(1, Q.H) = [c | B]
    """
    m, n = B.shape
    B_work = B.astype(float).copy()
    # 将 c 转换为明确的列向量，方便隐式矩阵运算
    c_work = np.asarray(c, dtype=float).flatten()[:, np.newaxis]

    P = np.eye(m)
    Q = np.eye(n)

    # 1. 初始左变换 P1 处理向量 c (隐式左乘)
    v_p1 = _householder_vec(c_work[:, 0])
    if np.any(v_p1):
        v_p1_2d = v_p1[:, np.newaxis]
        # H @ A = A - 2 * v * (v^T @ A)
        c_work -= 2.0 * v_p1_2d @ (v_p1_2d.T @ c_work)
        B_work -= 2.0 * v_p1_2d @ (v_p1_2d.T @ B_work)
        # 累积 P: P = P @ H1 (隐式右乘)
        P -= 2.0 * (P @ v_p1_2d) @ v_p1_2d.T

    for j in range(n):
        # 2. 右变换 Qj 处理行 (产生 beta)
        if j < n:
            row_to_zap = B_work[j, j:]
            v_q = _householder_vec(row_to_zap)
            if np.any(v_q):
                v_q_2d = v_q[:, np.newaxis]
                # B @ H = B - 2 * (B @ v) * v^T (隐式右乘)
                # 因为前面的列已经是 0，只需作用于 B_work[:, j:]
                B_work[:, j:] -= 2.0 * (B_work[:, j:] @ v_q_2d) @ v_q_2d.T
                # 累积 Q = Q @ H_q_full
                Q[:, j:] -= 2.0 * (Q[:, j:] @ v_q_2d) @ v_q_2d.T

        # 3. 左变换 Pj+1 处理列 (产生 gamma)
        if j < m - 1:
            col_to_zap = B_work[j + 1 :, j]
            v_p = _householder_vec(col_to_zap)
            if np.any(v_p):
                v_p_2d = v_p[:, np.newaxis]
                # 隐式左乘作用于子矩阵 B_work[j+1:, j:] 和 c_work[j+1:, :]
                B_work[j + 1 :, j:] -= 2.0 * v_p_2d @ (v_p_2d.T @ B_work[j + 1 :, j:])
                c_work[j + 1 :, :] -= 2.0 * v_p_2d @ (v_p_2d.T @ c_work[j + 1 :, :])
                # 累积 P = P @ H_p_full
                P[:, j + 1 :] -= 2.0 * (P[:, j + 1 :] @ v_p_2d) @ v_p_2d.T

    # 最终的 [c_tilde | B_tilde]，注意把 c 压平回 1D
    core_matrix = np.column_stack((c_work.flatten(), B_work))

    return P, core_matrix, Q


def core_problem_total_least_squares(
    A: NDArray_2D, b: NDArray_1D | NDArray_2D, p: int | None = None
) -> NDArray_2D:
    _, core_matrix, Q = _transform_to_core_problem(A, b)
    if p is None:
        p = A.shape[1]
    elif p > A.shape[1]:
        raise ValueError("p must be less than or equal to A.shape[1]")
    else:
        pass
    core_p = core_matrix[: p + 1, : p + 1]  # type: ignore
    x = total_least_squares(core_p[:, 1:], core_p[:, 0])

    return Q[:, : x.shape[0]] @ x


ols = ordinary_least_squares
tls = total_least_squares
ttls = truncated_total_least_squares
cptls = core_problem_total_least_squares
