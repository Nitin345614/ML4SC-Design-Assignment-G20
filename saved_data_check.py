import numpy as np
import matplotlib.pyplot as plt

data = np.load('Q-checkpoint-setup-train-1st-working_3.npz')
# data = np.load('DQN-simulation-data.npz')

Y = data['Y']
sin_theta = Y[:,0]
cos_theta = Y[:,1]
w = Y[:,2]

u = data['U']

theta = np.abs(np.arctan2(sin_theta, cos_theta))

fig, axs = plt.subplots(4, 1, figsize=(8, 10))
axs[0].plot(sin_theta)
axs[0].set_ylabel(f'$sin(\\Theta)$')
axs[0].grid()
axs[1].plot(cos_theta)
axs[1].set_ylabel(f'$cos(\\Theta)$')
axs[1].grid()
axs[2].plot(w)
axs[2].set_ylabel(f'$\\omega$')
axs[2].grid()
axs[3].plot(u)
axs[3].set_ylabel('control input $u$')
axs[3].set_xlabel('time step')
plt.show()

# fig, axs = plt.subplots(3, 1, figsize=(8, 10))
# axs[0].plot(theta)
# axs[0].set_ylabel(f'$\\Theta$')
# axs[1].plot(w)
# axs[1].set_ylabel(f'$\\omega$')
# axs[1].grid()
# axs[2].plot(u)
# axs[2].set_ylabel('control input $u$')
# axs[2].set_xlabel('time step')
# plt.show()