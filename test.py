import numpy as np
import matplotlib.pyplot as plt

batch_size = 12800
z_dim = 2
z_real_dist = np.random.randn(batch_size, z_dim)*5

plt.figure(figsize=(6,6))
plt.scatter(z_real_dist[:,0],z_real_dist[:,1])
# plt.colorbar()
plt.show()