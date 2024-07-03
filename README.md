# prophet

Analytic implementation of Facebook Prophet.

Prophet's a time series forecasting algorithm published by Facebook Core Data Science Team back in 2017. It gained popularity since then and several companies reported using it.

A successor - NeuralProphet - has been made in 2021, but it's out of the scope of this repo :)

Prophet model is a Generalized Additive Model (GAM) composed of 3 components, is defined as follows


$ \displaystyle
y(t) = g(t) + s(t) + h(t) + \epsilon_t \\
g(t) = (k + a(t)^T\delta)t + (m+a(t)^T\gamma) \\
s(t) = \sum_n a_n \cos(\frac{2\pi tn}{P}) + b_n \sin(\frac{2\pi tn}{P}) = X(t)^T\beta \\
h(t) = \sum_i \kappa_i \mathcal{1}_{t \in D_i} = Z(t)^T\kappa
$

where $g, s, h$ represents respectively the trend, seasonality and holidays components, further details are provided in the Prophet official paper.

Prophet's trained udner Maximum A Posteriori - meaning priors are declared on the parameters, likelihood is declared on the data, and a function - posterior distribution - is now well-defined, using Bayes formula.

$\mathcal{L}_{k, m, \delta, \beta, \kappa}(y) \pi(k, m, \delta, \beta, \kappa)$