#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>

double det_dot(const std::vector<double>& a, const std::vector<std::vector<double>>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i][0];
    }
    return sum;
}

std::vector<std::vector<double>> fourier_components(const std::vector<double>& t_days, double period, int n) {
    std::vector<std::vector<double>> x(t_days.size(), std::vector<double>(2 * n));
    for (size_t i = 0; i < t_days.size(); ++i) {
        for (int j = 0; j < n; ++j) {
            double freq = 2 * M_PI * (j + 1) / period;
            x[i][j] = cos(freq * t_days[i]);
            x[i][n + j] = sin(freq * t_days[i]);
        }
    }
    return x;
}

std::tuple<double, double, std::vector<double>, std::vector<double>> extract_params(const std::vector<double>& params) {
    double k = params[0];
    double m = params[1];
    std::vector<double> delta(params.begin() + 2, params.begin() + 27);
    std::vector<double> beta(params.begin() + 27, params.begin() + 47);
    return std::make_tuple(k, m, delta, beta);
}

double minus_log_posterior(const std::vector<double>& params, 
                           const std::vector<double>& t_scaled, 
                           const std::vector<double>& normalized_y, 
                           const std::vector<double>& change_points, 
                           double tau, double sigma, double sigma_obs, 
                           double sigma_k, double sigma_m, double period) {

    auto [k, m, delta, beta] = extract_params(params);
    std::vector<std::vector<double>> A(t_scaled.size(), std::vector<double>(change_points.size(), 0.0));
    for (size_t i = 0; i < t_scaled.size(); ++i) {
        for (size_t j = 0; j < change_points.size(); ++j) {
            A[i][j] = t_scaled[i] > change_points[j] ? 1.0 : 0.0;
        }
    }

    std::vector<double> gamma(change_points.size());
    for (size_t i = 0; i < change_points.size(); ++i) {
        gamma[i] = -change_points[i] * delta[i];
    }

    std::vector<double> g(t_scaled.size(), m);
    for (size_t i = 0; i < t_scaled.size(); ++i) {
        for (size_t j = 0; j < change_points.size(); ++j) {
            g[i] += A[i][j] * (delta[j] * t_scaled[i] + gamma[j]);
        }
        g[i] += k * t_scaled[i];
    }

    auto x = fourier_components(t_scaled, period, 10);
    std::vector<double> s(t_scaled.size(), 0.0);
    for (size_t i = 0; i < t_scaled.size(); ++i) {
        for (size_t j = 0; j < beta.size(); ++j) {
            s[i] += x[i][j] * beta[j];
        }
    }

    std::vector<double> y_pred(t_scaled.size());
    for (size_t i = 0; i < t_scaled.size(); ++i) {
        y_pred[i] = g[i] + s[i];
    }

    double minus_log_posterior = 0.0;
    for (size_t i = 0; i < normalized_y.size(); ++i) {
        minus_log_posterior += std::pow(normalized_y[i] - y_pred[i], 2) / (2 * std::pow(sigma_obs, 2));
    }

    minus_log_posterior += std::pow(k, 2) / (2 * std::pow(sigma_k, 2));
    minus_log_posterior += std::pow(m, 2) / (2 * std::pow(sigma_m, 2));
    for (const auto& b : beta) {
        minus_log_posterior += std::pow(b, 2) / (2 * std::pow(sigma, 2));
    }
    for (const auto& d : delta) {
        minus_log_posterior += std::abs(d) / tau;
    }

    return minus_log_posterior;
}

// Similar functions for computing gradient...