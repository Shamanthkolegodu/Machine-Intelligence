import numpy as np


class HMM:

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(
            zip(self.emissions, list(range(self.M))))

    def viterbi_algorithm(self, seq):
        seq_len = len(seq)
        num123 = np.zeros((seq_len, self.N))
        attr = np.zeros((seq_len, self.N), dtype=int)
        for j in range(self.N):
            num123[0, j] = self.pi[j] * self.B[j, self.emissions_dict[seq[0]]]
            attr[0, j] = 0
        for i in range(1, seq_len):
            for j in range(self.N):
                nu_max = -1
                maxOfTemp = -1
                for k in range(self.N):
                    localNu = num123[i - 1, k] * self.A[k, j] * \
                        self.B[j, self.emissions_dict[seq[i]]]
                    if localNu > nu_max:
                        nu_max = localNu
                        maxOfTemp = k
                num123[i, j] = nu_max
                attr[i, j] = maxOfTemp
        nu_max = -1
        maxOfTemp = -1
        for j in range(self.N):
            localNu = num123[seq_len - 1, j]
            if localNu > nu_max:
                nu_max = localNu
                maxOfTemp = j
        states = [maxOfTemp]
        for i in range(seq_len - 1, 0, -1):
            states.append(attr[i, states[-1]])
        states.reverse()

        self.states_dict = {v: k for k, v in self.states_dict.items()}
        return [self.states_dict[i] for i in states]
