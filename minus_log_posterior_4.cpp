#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <chrono>  // For timing

Eigen::MatrixXd fourier_components(const Eigen::VectorXd& t_days, double period, int n) {
    Eigen::MatrixXd x(t_days.size(), 2 * n);
    double factor = 2 * M_PI / period;
    for (int i = 0; i < t_days.size(); ++i) {
        for (int j = 1; j <= n; ++j) {
            double angle = factor * j * t_days[i];
            x(i, 2 * j - 2) = std::cos(angle);
            x(i, 2 * j - 1) = std::sin(angle);
        }
    }
    return x;
}

std::tuple<double, double, Eigen::VectorXd, Eigen::VectorXd> extract_params(const Eigen::VectorXd& params) {
    double k = params(0);
    double m = params(1);
    Eigen::VectorXd delta = params.segment(2, 25);
    Eigen::VectorXd beta = params.tail(params.size() - 27);
    return std::make_tuple(k, m, delta, beta);
}

double minus_log_posterior(const Eigen::VectorXd& params,
                           const Eigen::VectorXd& t_scaled,
                           const Eigen::VectorXd& change_points,
                           double scale_period,
                           const Eigen::VectorXd& normalized_y,
                           double sigma_obs,
                           double sigma_k,
                           double sigma_m,
                           double sigma,
                           double tau) {
    double k, m;
    Eigen::VectorXd delta, beta;
    std::tie(k, m, delta, beta) = extract_params(params);

    // Trend component
    Eigen::MatrixXd A = (t_scaled.replicate(1, change_points.size()).array() > change_points.transpose().replicate(t_scaled.size(), 1).array()).cast<double>();

    Eigen::VectorXd gamma = -delta.array() * change_points.array();
    Eigen::VectorXd g = k * t_scaled.array() + m;
    for (int i = 0; i < change_points.size(); ++i) {
        //g += (t_scaled.array() > change_points(i)).select(delta(i) * t_scaled.array() + gamma(i), 0);
        g += ((t_scaled.array() > change_points(i)).select(delta(i) * t_scaled.array() + gamma(i), 0)).matrix();
    }

    // Seasonality component
    double period = 365.25 / scale_period;
    Eigen::MatrixXd x = fourier_components(t_scaled, period, 10);
    Eigen::VectorXd s = x * beta;

    Eigen::VectorXd y_pred = g + s;

    double sum_squared_diff = (normalized_y - y_pred).array().square().sum();

    double minus_log_posterior = sum_squared_diff / (2 * std::pow(sigma_obs, 2)) + 
                                 std::pow(k, 2) / (2 * std::pow(sigma_k, 2)) + 
                                 std::pow(m, 2) / (2 * std::pow(sigma_m, 2)) + 
                                 beta.array().square().sum() / (2 * std::pow(sigma, 2)) + 
                                 delta.array().abs().sum() / tau;

    return minus_log_posterior;
}

extern "C" {
    double timed_minus_log_posterior(const double* params,
                                     int params_size,
                                     const double* t_scaled,
                                     int t_scaled_size,
                                     const double* change_points,
                                     int change_points_size,
                                     double scale_period,
                                     const double* normalized_y,
                                     int normalized_y_size,
                                     double sigma_obs,
                                     double sigma_k,
                                     double sigma_m,
                                     double sigma,
                                     double tau,
                                     int num_runs);
}

double timed_minus_log_posterior(const double* params,
                                 int params_size,
                                 const double* t_scaled,
                                 int t_scaled_size,
                                 const double* change_points,
                                 int change_points_size,
                                 double scale_period,
                                 const double* normalized_y,
                                 int normalized_y_size,
                                 double sigma_obs,
                                 double sigma_k,
                                 double sigma_m,
                                 double sigma,
                                 double tau,
                                 int num_runs) {
    Eigen::Map<const Eigen::VectorXd> params_vec(params, params_size);
    Eigen::Map<const Eigen::VectorXd> t_scaled_vec(t_scaled, t_scaled_size);
    Eigen::Map<const Eigen::VectorXd> change_points_vec(change_points, change_points_size);
    Eigen::Map<const Eigen::VectorXd> normalized_y_vec(normalized_y, normalized_y_size);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        minus_log_posterior(params_vec,
                            t_scaled_vec,
                            change_points_vec,
                            scale_period,
                            normalized_y_vec,
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
// g++ -std=c++17 -shared -fPIC -O3 -o libminus_log_posterior_4.so minus_log_posterior_4.cpp -I/opt/homebrew/opt/eigen/include/eigen3