import numpy as np
batch_size = 4
z_dim = 2
z_real_dist = np.random.randn(batch_size, z_dim)
print(z_real_dist)
print(z_real_dist*5)