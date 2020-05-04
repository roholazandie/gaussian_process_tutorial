import numpy as np

class GaussianProcessAnimation:

    def __init__(self, kernel, n_dims=150, n_frames=100):
        self.kernel = kernel
        self.n_dims = n_dims
        self.n_frames = n_frames


    def get_traces(self, n_traces=1):
        L = np.linalg.cholesky(self.kernel + 1.0e-8 * np.eye(len(self.kernel)))

        traces = []
        for _ in range(n_traces):
            s = self.gp_animation(self.n_dims, self.n_frames)
            frames = [np.dot(L.T, s[:, f]) for f in range(s.shape[1])]
            traces.append(frames)

        return traces


    def gp_animation(self, n_dim, n_frames):
        x = np.random.randn(n_dim, 1)
        r = np.sqrt(np.sum(x**2))
        x = x/r
        t = np.random.randn(n_dim, 1)
        t = t - np.dot(t.T, x) * x
        t = t/np.sqrt(np.sum(t**2))
        s = np.linspace(0, 2*np.pi, n_frames)
        #s = s[0: -2]
        t = np.dot(t, s[np.newaxis])
        X = r * self._exp_map(x, t)
        return X


    def _exp_map(self, mu, E):
        n_dim = E.shape[0]
        theta = np.sqrt(np.sum(E**2, axis=0))
        M = mu * np.cos(theta) + E * np.tile(np.sin(theta)/theta, (n_dim, 1))
        if np.any(np.abs(theta) < 1e-7):
            for i in np.where(np.abs(theta) < 1e-7)[0]:
                M[:, i] = mu.ravel()

        return M



if __name__ == "__main__":
    gp_animation = GaussianProcessAnimation(2)
    gp_animation.gp_animation(5, 10)