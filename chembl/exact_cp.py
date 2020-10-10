"""Implementation of exact conformal prediction for ridge regression.

From https://github.com/EugeneNdiaye/homotopy_conformal_prediction.
"""

import numpy as np
import intervals


def conf_pred(X, Y_, lambda_, alpha=0.1):

    n_samples, n_features = X.shape
    H = X.T.dot(X) + lambda_ * np.eye(n_features)
    C = np.eye(n_samples) - X.dot(np.linalg.solve(H, X.T))
    A = C.dot(list(Y_) + [0])
    B = C[:, -1]

    negative_B = np.where(B < 0)[0]
    A[negative_B] *= -1
    B[negative_B] *= -1
    S, U, V = [], [], []

    for i in range(n_samples):

        if B[i] != B[-1]:
            tmp_u_i = (A[i] - A[-1]) / (B[-1] - B[i])
            tmp_v_i = -(A[i] + A[-1]) / (B[-1] + B[i])
            u_i, v_i = np.sort([tmp_u_i, tmp_v_i])
            U += [u_i]
            V += [v_i]

        elif B[i] != 0:
            tmp_uv = -0.5 * (A[i] + A[-1]) / B[i]
            U += [tmp_uv]
            V += [tmp_uv]

        if B[-1] > B[i]:
            S += [intervals.closed(U[i], V[i])]

        elif B[-1] < B[i]:
            intvl_u = intervals.openclosed(-np.inf, U[i])
            intvl_v = intervals.closedopen(V[i], np.inf)
            S += [intvl_u.union(intvl_v)]

        elif B[-1] == B[i] and B[i] > 0 and A[-1] < A[i]:
            S += [intervals.closedopen(U[i], np.inf)]

        elif B[-1] == B[i] and B[i] > 0 and A[-1] > A[i]:
            S += [intervals.openclosed(-np.inf, U[i])]

        elif B[-1] == B[i] and B[i] == 0 and abs(A[-1]) <= abs(A[i]):
            S += [intervals.open(-np.inf, np.inf)]

        elif B[-1] == B[i] and B[i] == 0 and abs(A[-1]) > abs(A[i]):
            S += [intervals.empty()]

        elif B[-1] == B[i] and A[-1] == A[i]:
            S += [intervals.open(-np.inf, np.inf)]

        else:
            print("boom !!!")

    hat_y = np.sort([-np.inf] + U + V + [np.inf])
    size = hat_y.shape[0]
    conf_pred = intervals.empty()

    for i in range(size - 1):

        n_pvalue_i = 0.
        intvl_i = intervals.closed(hat_y[i], hat_y[i + 1])
        for j in range(n_samples):
            n_pvalue_i += intvl_i in S[j]

        if n_pvalue_i > alpha * n_samples:
            conf_pred = conf_pred.union(intvl_i)

    return (conf_pred.lower, conf_pred.upper)
