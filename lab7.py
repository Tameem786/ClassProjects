import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1)
y1 = -np.log(x)
y2 = -np.log(1-x)

plt.plot(x, y1)
plt.plot(x, y2)
plt.show()
