import numpy as np
import myBellman as mB


def main():

    A = np.array([[
        [0, 0, 0.5],
        [1, 0, 0],
        [0, 1, 0.5]
    ]])

    R = np.array([[
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 1]
    ]])

    Pol = np.array([[1, 1, 1]])

    Pol = mB.pol_iterate(A, R, Pol, 0.99, 0.001)

    V = mB.pol_eval(A, R, Pol, 1, 0.5)
    print(Pol)
    print('\n')
    print(V)
    print('\n\n')

    Pol, V = mB.val_iterate(A, R, 1, 0.5)
    print(Pol)
    print('\n')
    print(V)
    print('\n\n')
    return
    return


if __name__ == '__main__':
    main()
