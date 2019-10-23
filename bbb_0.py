import numpy as np
import sys


def runTrace(pos, trace, labels, R, P, rew):
    while True:
        rew = rew + R[pos]
        pos = np.random.choice(7, p=P[pos])
        trace.append(labels[pos])
        if P[pos][pos] == 1:
            break
    print("Started in state {} \nEnded in state {} \nAccumulated {} reward\n".format(sys.argv[1], pos, rew))
    for i, lab in enumerate(trace):
        print(lab, end='')
        if i != len(trace) - 1:
            print(" -> ", end='')
    return


def main():
    s = np.zeros(7)
    if len(sys.argv) == 2:
        argin = int(sys.argv[1])
        s[argin] = 1
        pos = int(sys.argv[1])
    # 0: Facebook, 1:Class 1, 2: Class 2, 3: Class 3, 4: Pub, 5: Sleep, 6: Pass
    P = np.array([[0.9, 0.1, 0, 0, 0, 0, 0],
                  [0.5, 0, 0.5, 0, 0, 0, 0],
                  [0, 0, 0, 0.8, 0, 0.2, 0],
                  [0, 0, 0, 0, 0.4, 0, 0.6],
                  [0, 0.2, 0.4, 0.4, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1, 0]])
    labels = ["Facebook", "Class 1", "Class 2", "Class 3", "Pub", "Sleep", "Pass"]
    R = np.array([-1, -2, -2, -2, 1, 0, 10])
    rew = 0
    trace = list([labels[pos]])

    if len(sys.argv)== 2:
        runTrace(pos, trace, labels, R, P, rew)


if __name__ == "__main__":
    main()
