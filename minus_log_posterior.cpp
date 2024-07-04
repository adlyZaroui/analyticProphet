#include <cmath>
#include <vector>
#include <numeric>
#include <tuple>
#include <algorithm>
#include <iostream> // For debug output

std::vector<std::vector<double>> fourier_components(const std::vector<double>& t_days, double period, int n) {
    std::vector<std::vector<double>> x(t_days.size(), std::vector<double>(2 * n));
    for (size_t i = 0; i < t_days.size(); ++i) {
        for (int j = 1; j <= n; ++j) {
            x[i][2 * j - 2] = std::cos(2 * M_PI * j * t_days[i] / period);
            x[i][2 * j - 1] = std::sin(2 * M_PI * j * t_days[i] / period);
        }
    }
    return x;
}

std::tuple<double, double, std::vector<double>, std::vector<double>> extract_params(const std::vector<double>& params) {
    double k = params[0];
    double m = params[1];
    std::vector<double> delta(params.begin() + 2, params.begin() + 27);
    std::vector<double> beta(params.begin() + 27, params.end());
    return std::make_tuple(k, m, delta, beta);
}

extern "C" {
    double minus_log_posterior(const double* params,
                               size_t params_size,
                               const double* t_scaled,
                               size_t t_scaled_size,
                               const double* change_points,
                               size_t change_points_size,
                               double scale_period,
                               const double* normalized_y,
                               size_t normalized_y_size,
                               double sigma_obs,
                               double sigma_k,
                               double sigma_m,
                               double sigma,
                               double tau);
}

double minus_log_posterior(const double* params,
                           size_t params_size,
                           const double* t_scaled,
                           size_t t_scaled_size,
                           const double* change_points,
                           size_t change_points_size,
                           double scale_period,
                           const double* normalized_y,
                           size_t normalized_y_size,
                           double sigma_obs,
                           double sigma_k,
                           double sigma_m,
                           double sigma,
                           double tau) {
    // Convert input arrays to vectors
    std::vector<double> params_vec(params, params + params_size);
    std::vector<double> t_scaled_vec(t_scaled, t_scaled + t_scaled_size);
    std::vector<double> change_points_vec(change_points, change_points + change_points_size);
    std::vector<double> normalized_y_vec(normalized_y, normalized_y + normalized_y_size);

    auto [k, m, delta, beta] = extract_params(params_vec);

    // Debug output
    std::cout << "k: " << k << ", m: " << m << std::endl;
    std::cout << "delta size: " << delta.size() << ", beta size: " << beta.size() << std::endl;
    std::cout << "t_scaled size: " << t_scaled_vec.size() << ", change_points size: " << change_points_vec.size() << std::endl;

    // Trend component
    std::vector<std::vector<int>> A(t_scaled_vec.size(), std::vector<int>(change_points_vec.size(), 0));
    for (size_t i = 0; i < t_scaled_vec.size(); ++i) {
        for (size_t j = 0; j < change_points_vec.size(); ++j) {
            A[i][j] = t_scaled_vec[i] > change_points_vec[j] ? 1 : 0;
        }
    }

    std::vector<double> gamma(change_points_vec.size());
    for (size_t i = 0; i < change_points_vec.size(); ++i) {
        gamma[i] = -change_points_vec[i] * delta[i];
    }

    std::vector<double> g(t_scaled_vec.size());
    for (size_t i = 0; i < t_scaled_vec.size(); ++i) {
        g[i] = k * t_scaled_vec[i] + m;
        for (size_t j = 0; j < change_points_vec.size(); ++j) {
            g[i] += A[i][j] * (delta[j] * t_scaled_vec[i] + gamma[j]);
        }
    }

    // Seasonality component
    double period = 365.25 / scale_period;
    auto x = fourier_components(t_scaled_vec, period, 10);
    std::vector<double> s(x.size(), 0);
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < beta.size(); ++j) {
            s[i] += x[i][j] * beta[j];
        }
    }

    std::vector<double> y_pred(t_scaled_vec.size());
    for (size_t i = 0; i < t_scaled_vec.size(); ++i) {
        y_pred[i] = g[i] + s[i];
    }

    double sum_squared_diff = 0;
    for (size_t i = 0; i < t_scaled_vec.size(); ++i) {
        sum_squared_diff += std::pow(normalized_y_vec[i] - y_pred[i], 2);
    }

    double minus_log_posterior = sum_squared_diff / (2 * std::pow(sigma_obs, 2)) + 
                                 std::pow(k, 2) / (2 * std::pow(sigma_k, 2)) + 
                                 std::pow(m, 2) / (2 * std::pow(sigma_m, 2)) + 
                                 std::accumulate(beta.begin(), beta.end(), 0.0, [](double acc, double b) { return acc + std::pow(b, 2); }) / (2 * std::pow(sigma, 2)) + 
                                 std::accumulate(delta.begin(), delta.end(), 0.0, [](double acc, double d) { return acc + std::abs(d); }) / tau;

    return minus_log_posterior;
}