import numpy as np


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
        print(V)
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


def val_iterate(P, R, gamma, lim_delta):
    delta = 0
    V = np.zeros(P.shape[1])
    _V = np.array(V)
    v = np.zeros(P.shape[0])
    while True:
        _V = np.array(V)
        for s in range(P.shape[1]):
            for a in range(len(v)):
                v[a] = np.dot(P[a, :, s], (R[a, :, s] + gamma * _V))
            V[s] = np.max(v)
        delta = np.linalg.norm(_V - V)
        if delta < lim_delta:
            break
    Pol = np.array(np.zeros((len(v), len(V))))
    for s in range(Pol.shape[1]):
        decision = np.dot(P[:, :, s], (R[:, :, s] + gamma * V).transpose())
        Pol[np.argmax(np.diagonal(decision)), s] = 1
    return Pol, V
