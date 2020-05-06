import numpy as np
import pods
from GPy.models import GPRegression
from GPy.kern import RBF, Matern32
from visualization import model_output, plot_gp

'''
here we study the effect of lengthscale and variance on the resulting regression
lengthscale is very sensitive to different values best values are around 20
lower variance decreases the noise 
'''

data = pods.datasets.della_gatta_TRP63_gene_expression(data_set='della_gatta', gene_number=937)

x = data['X']
y = data['Y']

offset = np.mean(y)
scale = np.sqrt(np.var(y))

yhat = (y-offset)/scale

#kernel = RBF(input_dim=1, variance=100)
#kernel = Matern32(input_dim=1, variance=2.0, lengthscale=200)
model = GPRegression(x, yhat)
model.kern.lengthscale = 20 #this will widen with 100, 200
#gp_regression.likelihood.variance = 0.001

print(model.log_likelihood())
model.optimize()
print(model.log_likelihood())

xt = np.linspace(-20, 260, 100)[:, np.newaxis]
yt_mean, yt_var = model.predict(xt)

plot_gp(yt_mean, yt_var, xt, X_train=model.X.flatten(), Y_train=model.Y.flatten())

