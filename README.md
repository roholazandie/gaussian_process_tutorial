## Gaussian Process Tutorials
This repo contains codes and best practices to learn gp.

- a lot of codes are borrowed from Neil Lawrence works and codes


## GPytorch Models
Spectral Mixture for extrapolation in future


### Approximate models
In variational approximate models we are looking for a q dist that approximates p (the actual prior)
We want to minimize the ELBO(the first two terms of below) (we have to take gradients of ELBO which can be done with torch):
![](imgs/elbo.png)
Or similarly (because Y_i's are iid):
![](imgs/elbo2.png)
(In which Z is the inducting point set, D, Y are data points and target points (in training data) )

We can achive this by first choosing a gaussian family for q and then using 
``VariationalELBO`` (adds two terms of the above: KL term and likelihood term) Class with ``VariationalStrategy`` Class to minimize the negative of
 it (in gpytorch implementation the ELBO defined with negative sign of what we have here)
To find a relavant exmaple see: gpytorch_examples/variational_and_approximate_gps/approximate_gp.py



## Readings
The curated list of best reading on gp on the web:
- https://distill.pub/2019/visual-exploration-gaussian-processes/
- http://www.gaussianprocess.org/gpml/
- http://krasserm.github.io/2018/03/19/gaussian-processes/
## Demos
- http://www.tmpl.fi/gp/
- http://chifeng.scripts.mit.edu/stuff/gp-demo/