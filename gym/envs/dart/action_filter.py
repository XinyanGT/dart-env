from scipy import signal
import numpy as np

class ActionFilter:
    def __init__(self, dim, order, sampling_frequency, low_cutoff, high_cutoff):
        self.dim = dim
        self.order = order
        nyq = 0.5 * sampling_frequency
        low = low_cutoff / nyq + 1e-5
        high = high_cutoff / nyq

        b, a = signal.butter(order, [low, high], btype='band')
        self.filter = [b, a]


    def filter_action(self, action):
        if self.initial_action:
            for i in range(10):
                for d in range(self.dim):
                    _, self.z[d] = signal.lfilter(self.filter[0], self.filter[1], [action[d]], zi=self.z[d])
            self.initial_action = False
        filtered_action = np.copy(action)
        for d in range(self.dim):
            filtered_action[d], self.z[d] = signal.lfilter(self.filter[0], self.filter[1], [action[d]], zi=self.z[d])
        return filtered_action

    def reset_filter(self):
        z = signal.lfilter_zi(self.filter[0], self.filter[1])
        self.z = [z * 0 for i in range(self.dim)]
        self.initial_action = True
