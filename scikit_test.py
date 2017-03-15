import numpy as np
import matplotlib.pyplot as plt

data_path = r"C:/tmp/data.txt"
data = np.loadtxt(data_path)
# data = np.array([1, 2, 2, 5, 3, 4])
plt.plot(data, 'r')
plt.show()
print(np.median(data))
