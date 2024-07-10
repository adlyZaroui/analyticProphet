#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <iostream>

Eigen::MatrixXd fourier_components(const Eigen::VectorXd& t_days, double period, int n) {
    Eigen::MatrixXd x(t_days.size(), 2 * n);
    double factor = 2 * M_PI / period;
    for (int i = 0; i < t_days.size(); ++i) {
        for (int j = 0; j < n; ++j) { // Start j from 0 for 0-based indexing
            double angle = factor * (j + 1) * t_days[i]; // (j+1) because j starts from 0
            x(i, j) = std::cos(angle); // Fill first half with cosines
            x(i, j + n) = std::sin(angle); // Fill second half with sines
        }
    }
    return x;
}

std::tuple<double, double, Eigen::VectorXd, Eigen::VectorXd> extract_params(const Eigen::VectorXd& params) {
    double k = params(0);
    double m = params(1);
    Eigen::VectorXd delta = params.segment(2, 25); // Extract next 25 elements for delta
    Eigen::VectorXd beta = params.tail(params.size() - 27);
    //Eigen::VectorXd beta = params.segment(27, 20); // Ensure beta is extracted with exactly 20 elements
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

    // Declare grad vector
    Eigen::VectorXd grad(params_size);

    // Compute dk and dm
    grad(0) = -r.dot(t_scaled_vec) / (sigma_obs * sigma_obs) + k / (sigma_k * sigma_k);
    grad(1) = -r.sum() / (sigma_obs * sigma_obs) + m / (sigma_m * sigma_m);

    // Compute ddelta
    Eigen::MatrixXd t_diff = t_scaled_vec.replicate(1, change_points_vec.size()).array().rowwise() - change_points_vec.transpose().array();
    Eigen::MatrixXd delta_contrib = t_diff.array() * A.array();
    Eigen::VectorXd ddelta = -(r.transpose() * delta_contrib).transpose() / (sigma_obs * sigma_obs) + (delta.array().sign() / tau).matrix();
    
    grad.segment(2, delta.size()) = ddelta;

    // Compute dbeta
    int beta_start_index = 2 + delta.size(); // Dynamically calculate the starting index for beta
    Eigen::VectorXd dbeta = -(x.transpose() * r) / (sigma_obs * sigma_obs) + beta / (sigma * sigma);

    grad.segment(beta_start_index, beta.size()) = dbeta;

    std::memcpy(grad_out, grad.data(), grad.size() * sizeof(double));
}

// To compile this file on Mac OS, run:
// g++ -std=c++17 -shared -fPIC -O3 -o libgradient.so gradient.cpp -I/opt/homebrew/opt/eigen/include/eigen3