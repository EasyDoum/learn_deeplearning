import matplotlib.pyplot as plt
import numpy as np


def array_plot():
    a = np.zeros([3, 2])
    print(a)
    a[0, 0] = 1
    a[0, 1] = 5
    a[1, 0] = 10.3
    a[2, 1] = 16
    a[2, 0] = 20.6666
    print(a)
    print(a[1,0])
    plt.imshow(a)
    plt.show()
