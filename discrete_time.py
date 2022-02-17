import numpy as np
import scipy.signal as sig
from matplotlib import pyplot as plt


class DiscreteTime:

    def impulse_signal(self, minimum, maximum):
        n = np.arange(minimum, maximum + 1)
        delta_sig = sig.unit_impulse(n.shape, idx='mid')
        return delta_sig

    def unit_step(self, minimum, maximum):
        n = np.arange(minimum, maximum + 1)
        self.unit_signal = np.ones(n.shape)
        self.unit_signal[n < 0] = 0
        return self.unit_signal

    def ramp_signal(self, minimum, maximum):
        n = np.arange(minimum, maximum + 1)
        ramp_step = n * self.unit_step(minimum, maximum)
        return ramp_step

    def exponential_decay(self, minimum, maximum, alpha):
        n = np.arange(minimum, maximum + 1)
        exp_decay = (alpha ** n) * self.unit_step(minimum, maximum)
        return exp_decay

    def exponential_growth(self, minimum, maximum, alpha):
        n = np.arange(minimum, maximum + 1)
        exp_growth = (n ** alpha) * self.unit_step(minimum, maximum)
        exp_growth[np.isnan(exp_growth)] = 0
        return exp_growth

    def leaky_relu(self, minimum, maximum, alpha):
        n = np.arange(minimum, maximum + 1)
        l_relu = n * np.ones(n.shape)
        l_relu[n < 0] *= alpha
        return l_relu

    def elu(self, minimum, maximum, alpha):
        n = np.arange(minimum, maximum + 1)
        z = np.exp(n)
        unit_signal = np.ones(n.shape)
        unit_signal[n < 0] = z[z < 1] * alpha - 1
        unit_signal[n >= 0] = n[n >= 0]
        return unit_signal

    def sigmoid(self, minimum, maximum):
        n = np.arange(minimum, maximum + 1)
        sigmoid_signal = 1 / (1 + np.exp(-n))
        return sigmoid_signal

    def disc_plot(self, x, y, title, func_name='f[n]'):
        plt.figure(figsize=(8, 5))
        plt.title(title)
        plt.xlabel("n")
        plt.ylabel(func_name)
        plt.stem(x, y, use_line_collection=True)
        plt.grid()
        plt.show()

    def plot_signal(self, signal_chosen, minimum, maximum, alpha=0.5):
        n = np.arange(minimum, maximum + 1)
        discretetime = DiscreteTime()
        signals = {
            "impulse": discretetime.impulse_signal(minimum, maximum),
            "unit step": discretetime.unit_step(minimum, maximum),
            "ramp": discretetime.ramp_signal(minimum, maximum),
            "exponential decay": discretetime.exponential_decay(minimum, maximum, alpha),
            "exponential growth": discretetime.exponential_growth(minimum, maximum, alpha),
            "leaky relu": discretetime.leaky_relu(minimum, maximum, alpha),
            "eLU": discretetime.elu(minimum, maximum, alpha),
            "sigmoid": discretetime.sigmoid(minimum, maximum)
        }
        signal_plotter = signals.get(signal_chosen)

        return discretetime.disc_plot(n, signal_plotter, str(signal_chosen).title() + " Signal")
