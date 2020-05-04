import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gp_animation import GaussianProcessAnimation
from kernels import rbf_kernel


fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [])

def init():
    ax.set_xlim(-8, 8)
    ax.set_ylim(-5, 5)
    return ln,

def update(frame):
    xdata = np.linspace(-8, 8, n_dims).reshape(-1, 1)
    ydata = frame
    ln.set_data(xdata, ydata)
    return ln,


n_dims = 150
n_frames = 100
x = np.linspace(-8, 8, n_dims).reshape(-1, 1)
kernel = rbf_kernel(x, x)
gaussian_process_animation = GaussianProcessAnimation(kernel, n_dims=n_dims, n_frames=n_frames)
frames = gaussian_process_animation.get_traces(1)
frames = frames[0]

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=100)
plt.show()