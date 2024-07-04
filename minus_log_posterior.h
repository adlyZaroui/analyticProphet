#ifndef MINUS_LOG_POSTERIOR_H
#define MINUS_LOG_POSTERIOR_H

#include <vector>
#include <tuple>

std::vector<std::vector<double>> fourier_components(const std::vector<double>& t_days, double period, int n);
std::tuple<double, double, std::vector<double>, std::vector<double>> extract_params(const std::vector<double>& params);
double minus_log_posterior(const std::vector<double>& params,
                           const std::vector<double>& t_scaled,
                           const std::vector<double>& change_points,
                           double scale_period,
                           const std::vector<double>& normalized_y,
                           double sigma_obs,
                           double sigma_k,
                           double sigma_m,
                           double sigma,
                           double tau);

#endif // MINUS_LOG_POSTERIOR_H