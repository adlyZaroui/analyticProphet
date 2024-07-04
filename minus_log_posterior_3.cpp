#include <cmath>
#include <vector>
#include <numeric>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <chrono>  // For timing

std::vector<std::vector<double>> fourier_components(const std::vector<double>& t_days, double period, int n) {
    std::vector<std::vector<double>> x(t_days.size(), std::vector<double>(2 * n));
    double factor = 2 * M_PI / period;
    for (size_t i = 0; i < t_days.size(); ++i) {
        for (int j = 1; j <= n; ++j) {
            double angle = factor * j * t_days[i];
            x[i][2 * j - 2] = std::cos(angle);
            x[i][2 * j - 1] = std::sin(angle);
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

    double k, m;
    std::vector<double> delta, beta;
    std::tie(k, m, delta, beta) = extract_params(params_vec);

    // Trend component
    std::vector<std::vector<int>> A(t_scaled_vec.size(), std::vector<int>(change_points_vec.size()));
    std::transform(t_scaled_vec.begin(), t_scaled_vec.end(), A.begin(), [&](double t) {
        std::vector<int> a(change_points_vec.size());
        std::transform(change_points_vec.begin(), change_points_vec.end(), a.begin(), [t](double cp) {
            return t > cp ? 1 : 0;
        });
        return a;
    });

    std::vector<double> gamma(change_points_vec.size());
    std::transform(change_points_vec.begin(), change_points_vec.end(), delta.begin(), gamma.begin(), std::multiplies<double>());
    std::transform(gamma.begin(), gamma.end(), gamma.begin(), std::negate<double>());

    std::vector<double> g(t_scaled_vec.size(), m);
    std::transform(t_scaled_vec.begin(), t_scaled_vec.end(), g.begin(), [&](double t) {
        double trend = k * t + m;
        for (size_t j = 0; j < change_points_vec.size(); ++j) {
            trend += (t > change_points_vec[j]) ? (delta[j] * t + gamma[j]) : 0;
        }
        return trend;
    });

    // Seasonality component
    double period = 365.25 / scale_period;
    auto x = fourier_components(t_scaled_vec, period, 10);
    std::vector<double> s(x.size(), 0);
    for (size_t i = 0; i < x.size(); ++i) {
        s[i] = std::inner_product(x[i].begin(), x[i].end(), beta.begin(), 0.0);
    }

    std::vector<double> y_pred(t_scaled_vec.size());
    std::transform(g.begin(), g.end(), s.begin(), y_pred.begin(), std::plus<double>());

    double sum_squared_diff = std::inner_product(normalized_y_vec.begin(), normalized_y_vec.end(), y_pred.begin(), 0.0, std::plus<double>(), [](double y, double y_pred) {
        return std::pow(y - y_pred, 2);
    });

    double minus_log_posterior = sum_squared_diff / (2 * std::pow(sigma_obs, 2)) + 
                                 std::pow(k, 2) / (2 * std::pow(sigma_k, 2)) + 
                                 std::pow(m, 2) / (2 * std::pow(sigma_m, 2)) + 
                                 std::accumulate(beta.begin(), beta.end(), 0.0, [](double acc, double b) { return acc + std::pow(b, 2); }) / (2 * std::pow(sigma, 2)) + 
                                 std::accumulate(delta.begin(), delta.end(), 0.0, [](double acc, double d) { return acc + std::abs(d); }) / tau;

    return minus_log_posterior;
}

extern "C" {
    double timed_minus_log_posterior(const double* params,
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
                                     double tau,
                                     int num_runs);
}

double timed_minus_log_posterior(const double* params,
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
                                 double tau,
                                 int num_runs) {
    // Convert input arrays to vectors
    std::vector<double> params_vec(params, params + params_size);
    std::vector<double> t_scaled_vec(t_scaled, t_scaled + t_scaled_size);
    std::vector<double> change_points_vec(change_points, change_points + change_points_size);
    std::vector<double> normalized_y_vec(normalized_y, normalized_y + normalized_y_size);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        minus_log_posterior(params,
                            params_size,
                            t_scaled,
                            t_scaled_size,
                            change_points,
                            change_points_size,
                            scale_period,
                            normalized_y,
                            normalized_y_size,
                            sigma_obs,
                            sigma_k,
                            sigma_m,
                            sigma,
                            tau);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count();
}

// Compile with:
// g++ -std=c++17 -shared -fPIC -o libminus_log_posterior_3.so minus_log_posterior_3.cpp