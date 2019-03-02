import numpy as np
import matplotlib.pyplot as plt

T = 22709 * 9
# T = 200 * 391
T = int(T)
t = np.arange(1, T)
gamma = 1e-3
low = (1 - 1 / (gamma * t + 1))
high = (1 + 1 / (gamma * t))
plt.plot(t, low)
plt.plot(t, high)
plt.yscale('log')
# plt.show()

# for ind in [T//8, T*3//4, ]:
for ind in [T * 2 // 9, T * 5 // 9, T * 7 // 9]:
    print(high[ind] - 1)
