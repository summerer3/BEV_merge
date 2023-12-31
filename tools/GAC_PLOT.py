import matplotlib.pyplot as plt
import numpy as np

ps=np.array([[-0.6,2.25],
             [-0.6,4.3],
             [1.13,6.87],
             [2.13,6.87],
             [3.866,4.3],
             [3.866,2.25],
             [2.133,0.2],
             [1.13,0.2],
             [1.13,6.07],
             [2.13,6.07],
             [0,4.3],
             [0,2.25],
             [3.266,4.3],
             [3.266,2.25],
             [1.13,-0.6],
             [2.13,-0.6],
             [0,0],
             [0,6.27],
             [3.266,0],
             [3.266,6.27]])

print(ps.shape)
plt.scatter(ps[:,0],ps[:,1])
fig=plt.gca()
fig.set_aspect(1)
plt.show()