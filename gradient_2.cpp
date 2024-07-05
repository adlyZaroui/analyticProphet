#include <Eigen/Dense>
#include <cmath>
#include <tuple>

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

extern "C" {
    void gradient(const double* params,
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
                  double* grad_out);
}

void gradient(const double* params,
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
              double* grad_out) {
    Eigen::Map<const Eigen::VectorXd> params_vec(params, params_size);
    Eigen::Map<const Eigen::VectorXd> t_scaled_vec(t_scaled, t_scaled_size);
    Eigen::Map<const Eigen::VectorXd> change_points_vec(change_points, change_points_size);
    Eigen::Map<const Eigen::VectorXd> normalized_y_vec(normalized_y, normalized_y_size);

    double k, m;
    Eigen::VectorXd delta, beta;
    std::tie(k, m, delta, beta) = extract_params(params_vec);

    // Trend component
    Eigen::MatrixXd A = (t_scaled_vec.replicate(1, change_points_vec.size()).array() > change_points_vec.transpose().replicate(t_scaled_vec.size(), 1).array()).cast<double>();

    Eigen::VectorXd gamma = -delta.array() * change_points_vec.array();
    Eigen::VectorXd g = k * t_scaled_vec.array() + m;
    for (int i = 0; i < change_points_vec.size(); ++i) {
        g += ((t_scaled_vec.array() > change_points_vec(i)).select(delta(i) * t_scaled_vec.array() + gamma(i), 0)).matrix();
    }

    // Seasonality component
    double period = 365.25 / scale_period;
    Eigen::MatrixXd x = fourier_components(t_scaled_vec, period, 10);
    Eigen::VectorXd s = x * beta;

    Eigen::VectorXd r = normalized_y_vec - g - s;

    Eigen::VectorXd grad(params_size);
    grad(0) = -r.dot(t_scaled_vec) / (sigma_obs * sigma_obs) + k / (sigma_k * sigma_k);
    grad(1) = -r.sum() / (sigma_obs * sigma_obs) + m / (sigma_m * sigma_m);

    for (int i = 0; i < delta.size(); ++i) {
        grad(2 + i) = -((r.array() * (t_scaled_vec.array() > change_points_vec(i)).select(t_scaled_vec.array() - change_points_vec(i), 0)).sum()) / (sigma_obs * sigma_obs) + std::copysign(1.0, delta(i)) / tau;
    }

    Eigen::VectorXd dbeta = -(r.transpose() * x).transpose() / (sigma_obs * sigma_obs) + beta / (sigma * sigma);
    grad.tail(beta.size()) = dbeta;

    std::memcpy(grad_out, grad.data(), grad.size() * sizeof(double));
}

// To compile this file on Mac OS, run:
// g++ -std=c++17 -shared -fPIC -O3 -o libgradient.so gradient.cpp -I/opt/homebrew/opt/eigen/include/eigen3