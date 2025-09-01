import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt


A = 2
signal_func = lambda x: A * np.sin(2 * np.pi * x)

delta_x = 0.01
freq = 100 # hz
x = np.linspace(0, 1, freq)
y = signal_func(x)

noise = A*0.3 * rnd.uniform(-0.5, 0.5, freq)
y = y + noise

high_pass = [0.5, -0.5] # high pass filter
low_pass = [0.5, 0.5] # low pass filter
filter_h = lambda x, box: np.convolve(x, box, mode='same')

# other way (own implementation)
def filter_h_own(x, box):
    y = np.zeros_like(x)
    for i in range(1, len(x)-1):
        y[i] = box[0] * x[i] + box[1] * x[i-1]
    return y

y_high_pass = filter_h(y, high_pass)
y_low_pass = filter_h(y, low_pass)
y_high_pass_own = filter_h_own(y, high_pass)
y_low_pass_own = filter_h_own(y, low_pass)

plt.plot(x, y, label="signal")
plt.plot(x, y_high_pass, label="high pass")
plt.plot(x, y_low_pass, label="low pass")
plt.plot(x, y_high_pass_own, label="high pass own")
plt.plot(x, y_low_pass_own, label="low pass own")
plt.legend()
plt.show()