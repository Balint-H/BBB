import numpy as np


def pol_eval(P, R, Pol, gamma, lim_delta):
    V = np.zeros(Pol.shape[1])
    v = np.zeros(Pol.shape[0])
    while True:
        _V = np.array(V)
        delta = 0
        for s in range(Pol.shape[1]):
            if P[0,s,s] == 1:
                continue
            for a in range(len(Pol[:, s])):
                v[a] = np.dot(P[a, :, s], (R[a, :, s] + gamma * _V))
            V[s] = np.dot(v, Pol[:, s])
            delta = np.max((np.abs(_V[s]-V[s]), delta))
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
    V = np.zeros(P.shape[1])
    Q = np.zeros(P.shape[0])
    epoch = 0
    while True:
        epoch += 1
        delta = 0
        for s in range(P.shape[1]):
            v = V[s]
            if P[0,s,s] == 1:
                continue
            for a in range(len(Q)):
                Q[a] = np.dot(P[a, :, s], (R[a, :, s] + gamma * V))
            V[s] = np.max(Q)
            delta = np.max((delta, np.abs(V[s] - v)))
        if delta < lim_delta:
            break
    Pol = np.array(np.zeros((len(Q), len(V))))
    for s in range(Pol.shape[1]):
        decision = np.dot(P[:, :, s], (R[:, :, s] + gamma * V).transpose())
        Pol[np.argmax(np.diagonal(decision)), s] = 1
    return Pol, V, epoch