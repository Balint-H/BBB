import numpy as np
import pdb


def main():
    S = np.matrix([0, 0.7, 0.3])
    S = S.transpose()
    A = np.array([[
        [0, 0.9, 0.2], 
        [0.4, 0, 0.0], 
        [0.6, 0.1, 0.8]],
        [
        [0.1, 1, 0.4], 
        [0.2, 0, 0.3], 
        [0.7, 0.0, 0.3]]])
        
    R = np.array([[
        [0, 1, 6], 
        [-10, 0, 0.0], 
        [0, 0, 0]],
        [
        [4, 1, -2], 
        [0, 0, 0], 
        [5, -10, -8]]])
        
    Pol = np.array([[0.9, 0.8, 0.5], [0.1, 0.2, 0.5]])
    P = MDP_P(A, Pol, S)
    V = pol_eval(A, R, S, Pol, 0.8, 0.0001)

    return


# A[a][s][s'] Na*N*N, Pol[a][s] Na*N, S[s] N*1
def MDP_P(A, Pol, S):
    N = S.shape[0]
    P = np.zeros((N, N))

    for (i, a_P) in enumerate(A):
        _Pol = np.array([Pol[i, :], ] * N)
        P = P + np.multiply(_Pol, a_P)
    return P


# A[a][s][s'] Na*N*N, Pol[a][s] Na*N, S[s] N*1
def MDP_R(A, Pol, R):
    N = S.shape[0]
    R = np.zeros((N, N))
    for (i, a_P) in enumerate(A):
        _Pol = np.array([Pol[i, :], ] * N)
        _R = np.multiply(_Pol, a_P)
        R = R + np.multiply(_R, R[i,:,:])       
    return R


def pol_eval(P, R, S, Pol, gamma, lim_delta):
    delta = 0
    V = np.zeros(len(S))
    _V = np.array(V)
    v = np.zeros(Pol.shape[0])
    while True:
        for s in range(len(S)):
            _V = np.array(V)
            for (a, _Pol) in enumerate(Pol[:, s]):
                v[a] = _Pol * np.sum(np.dot(P[a, :, s], (R[a, :, s] + gamma * V)))
            V[s] = np.sum(v)
            delta = np.linalg.norm(_V - V)
            print(V)
        if delta < lim_delta:
            break
    return V  


def step(pos, Pol, Act, Rew):

    _act = np.random.choice(Pol.shape[1], p=Pol[:, pos])

    pos_prime = np.random.choice(Act.shape[2], p=Act[_act, pos, :])

    _rew = Rew[pos][pos_prime][_act]
    return (pos_prime, _rew)


if __name__ == '__main__':
    main()

