import numpy as np
import matplotlib.pyplot as plt

data = [
    1,
    0.33309453,
    0.10675623,
    0.00908689,
    -0.05878484,
    -0.09750013,
    -0.12087089,
    -0.1324179,
    -0.13696159,
    -0.13295039,
    -0.11765382,
]
x = np.arange(0, 11, 1)
plt.plot(x, data)
plt.xlabel("Regression lag")
plt.ylabel("Autocorrelation")
# add a horizontal line at y=0
plt.axhline(y=0, color="k")
plt.show()
