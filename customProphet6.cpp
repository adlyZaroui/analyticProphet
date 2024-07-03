#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <tuple>

extern "C" {
    double det_dot(const double* a, const double* b, size_t size) {
        double sum = 0.0;
        for (size_t i = 0; i < size; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    void fourier_components(const double* t_days, size_t t_days_size, double period, int n, double* out_x) {
        for (size_t i = 0; i < t_days_size; ++i) {
            for (int j = 0; j < n; ++j) {
                double freq = 2 * M_PI * (j + 1) / period;
                out_x[i * 2 * n + j] = cos(freq * t_days[i]);
                out_x[i * 2 * n + n + j] = sin(freq * t_days[i]);
            }
        }
    }

    void extract_params(const double* params, double* k, double* m, double* delta, double* beta) {
        *k = params[0];
        *m = params[1];
        std::copy(params + 2, params + 27, delta);
        std::copy(params + 27, params + 47, beta);
    }

    double minus_log_posterior(const double* params, const double* t_scaled, const double* normalized_y, 
                               const double* change_points, size_t t_scaled_size, size_t change_points_size,
                               double tau, double sigma, double sigma_obs, double sigma_k, double sigma_m, double period) {

        double k, m;
        std::vector<double> delta(25);
        std::vector<double> beta(20);
        extract_params(params, &k, &m, delta.data(), beta.data());

        std::vector<std::vector<double> > A(t_scaled_size, std::vector<double>(change_points_size, 0.0));
        for (size_t i = 0; i < t_scaled_size; ++i) {
            for (size_t j = 0; j < change_points_size; ++j) {
                A[i][j] = t_scaled[i] > change_points[j] ? 1.0 : 0.0;
            }
        }

        std::vector<double> gamma(change_points_size);
        for (size_t i = 0; i < change_points_size; ++i) {
            gamma[i] = -change_points[i] * delta[i];
        }

        std::vector<double> g(t_scaled_size, m);
        for (size_t i = 0; i < t_scaled_size; ++i) {
            for (size_t j = 0; j < change_points_size; ++j) {
                g[i] += A[i][j] * (delta[j] * t_scaled[i] + gamma[j]);
            }
            g[i] += k * t_scaled[i];
        }

        std::vector<double> x(t_scaled_size * 20);
        fourier_components(t_scaled, t_scaled_size, period, 10, x.data());

        std::vector<double> s(t_scaled_size, 0.0);
        for (size_t i = 0; i < t_scaled_size; ++i) {
            for (size_t j = 0; j < beta.size(); ++j) {
                s[i] += x[i * 20 + j] * beta[j];
            }
        }

        std::vector<double> y_pred(t_scaled_size);
        for (size_t i = 0; i < t_scaled_size; ++i) {
            y_pred[i] = g[i] + s[i];
        }

        double minus_log_posterior = 0.0;
        for (size_t i = 0; i < t_scaled_size; ++i) {
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
}