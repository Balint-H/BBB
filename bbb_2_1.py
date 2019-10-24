import numpy as np
import pdb

np.set_printoptions(precision=2)


def main():
    S = np.zeros(7)
    S = S.transpose()
    A = np.array([[
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1]],
        [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1]]])

    R = np.array([[
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 10, 0]],
        [
            [0, -10, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]]])

    Pol = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    # P = MDP_P(A, Pol)
    #
    # Pol, stable = pol_improve(A, R, Pol, V, 0.9)
    Pol = pol_iterate(A, R, Pol, 1, .0001)
    V = pol_eval(A, R, Pol, 1, 0.0001)
    print(Pol)
    print('\n')
    print(V)
    return


# A[a][s][s'] Na*N*N, Pol[a][s] Na*N, S[s] N*1
def MDP_P(A, Pol):
    N = Pol.shape[1]
    P = np.zeros((N, N))

    for (i, a_P) in enumerate(A):
        _Pol = np.array([Pol[i, :], ] * N)
        P = P + np.multiply(_Pol, a_P)
    return P


# A[a][s][s'] Na*N*N, Pol[a][s] Na*N, S[s] N*1
def MDP_R(A, Pol, R):
    N = Pol.shape[0]
    R = np.zeros((N, N))
    for (i, a_P) in enumerate(A):
        _Pol = np.array([Pol[i, :], ] * N)
        _R = np.multiply(_Pol, a_P)
        R = R + np.multiply(_R, R[i, :, :])
    return R


def pol_eval(P, R, Pol, gamma, lim_delta):
    delta = 0
    V = np.zeros(Pol.shape[1])
    _V = np.array(V)
    v = np.zeros(Pol.shape[0])
    while True:
        _V = np.array(V)
        for s in range(Pol.shape[1]):
            for a in range(len(Pol[:, s])):
                v[a] = np.dot(P[a, :, s], (R[a, :, s] + gamma * _V))
            V[s] = np.dot(v, Pol[:, s])
            delta = np.linalg.norm(_V - V)
        if delta < lim_delta:
            break
    return V


def pol_improve(P, R, Pol, V, gamma):
    stable = True
    _Pol = np.array(Pol)
    Pol = np.array(np.zeros(Pol.shape))
    for s in range(Pol.shape[1]):
        decision = np.dot(P[:, :, s], (R[:, :, s] + gamma * V).transpose())
        Pol[np.argmax(np.diagonal(decision)), s] = 1
    if not np.array_equal(_Pol, Pol):
        stable = False
    return Pol, stable


def pol_iterate(P, R, Pol, gamma, lim_delta):
    stable = False
    while not stable:
        V = pol_eval(P, R, Pol, gamma, lim_delta)
        Pol, stable = pol_improve(P, R, Pol, V, gamma)

    return Pol


def step(pos, Pol, Act, Rew):
    _act = np.random.choice(Pol.shape[1], p=Pol[:, pos])

    pos_prime = np.random.choice(Act.shape[2], p=Act[_act, pos, :])

    _rew = Rew[_act][pos][pos_prime]
    return pos_prime, _rew


if __name__ == '__main__':
    main()
