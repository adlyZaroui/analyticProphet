#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <iostream>
#include <lbfgs.h>
#include <cstring>

Eigen::MatrixXd fourier_components(const Eigen::VectorXd& t_days, double period, int n) {
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n, 1, n) * (2 * M_PI / period);
    Eigen::MatrixXd angles = t_days * x.transpose();
    
    Eigen::MatrixXd result(t_days.size(), 2 * n);
    result.leftCols(n) = angles.array().cos();
    result.rightCols(n) = angles.array().sin();
    
    return result;
}

std::tuple<double, double, Eigen::VectorXd, Eigen::VectorXd> extract_params(const Eigen::Ref<const Eigen::VectorXd>& params) {
    double k = params(0);
    double m = params(1);
    Eigen::VectorXd delta = params.segment(2, 25); // Extract next 25 elements for delta
    Eigen::VectorXd beta = params.tail(params.size() - 27);
    return std::make_tuple(k, m, delta, beta);
}

void minus_log_posterior_and_gradient(const Eigen::VectorXd& params_vec,
                                      const Eigen::VectorXd& t_scaled_vec,
                                      const Eigen::VectorXd& change_points_vec,
                                      double scale_period,
                                      const Eigen::VectorXd& normalized_y_vec,
                                      double sigma_obs,
                                      double sigma_k,
                                      double sigma_m,
                                      double sigma,
                                      double tau,
                                      double& mlp_out,
                                      Eigen::Ref<Eigen::VectorXd> grad_out) {
    double k, m;
    Eigen::VectorXd delta, beta;
    std::tie(k, m, delta, beta) = extract_params(params_vec);

    // Trend component
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(t_scaled_vec.size());
    Eigen::MatrixXd A = (t_scaled_vec.replicate(1, change_points_vec.size()).array() > change_points_vec.transpose().replicate(t_scaled_vec.size(), 1).array()).cast<double>();
    Eigen::VectorXd gamma = -delta.array() * change_points_vec.array();
    Eigen::VectorXd g = (k * ones + A * delta).array() * t_scaled_vec.array() + (m * ones + A * gamma).array();

    // Seasonality component
    double period = 365.25 / scale_period;
    Eigen::MatrixXd x = fourier_components(t_scaled_vec, period, 10);
    Eigen::VectorXd s = x * beta;

    Eigen::VectorXd y_pred = g + s;
    Eigen::VectorXd r = normalized_y_vec - y_pred;

    double sum_squared_diff = r.array().square().sum();

    double minus_log_posterior_value = sum_squared_diff / (2 * std::pow(sigma_obs, 2)) + 
                                       std::pow(k, 2) / (2 * std::pow(sigma_k, 2)) + 
                                       std::pow(m, 2) / (2 * std::pow(sigma_m, 2)) + 
                                       beta.array().square().sum() / (2 * std::pow(sigma, 2)) + 
                                       delta.array().abs().sum() / tau;

    // Set minus log posterior value
    mlp_out = minus_log_posterior_value;

    // Compute gradients
    grad_out.resize(params_vec.size());

    // Compute dk and dm
    grad_out(0) = -r.dot(t_scaled_vec) / (sigma_obs * sigma_obs) + k / (sigma_k * sigma_k);
    grad_out(1) = -r.sum() / (sigma_obs * sigma_obs) + m / (sigma_m * sigma_m);

    // Compute ddelta
    Eigen::MatrixXd t_diff = t_scaled_vec.replicate(1, change_points_vec.size()).array().rowwise() - change_points_vec.transpose().array();
    Eigen::MatrixXd delta_contrib = t_diff.array() * A.array();
    Eigen::VectorXd ddelta = -(r.transpose() * delta_contrib).transpose() / (sigma_obs * sigma_obs) + (delta.array().sign() / tau).matrix();
    
    grad_out.segment(2, delta.size()) = ddelta;

    // Compute dbeta
    int beta_start_index = 2 + delta.size(); // Dynamically calculate the starting index for beta
    Eigen::VectorXd dbeta = -(x.transpose() * r) / (sigma_obs * sigma_obs) + beta / (sigma * sigma);

    grad_out.segment(beta_start_index, beta.size()) = dbeta;
}

struct OptimizationData {
    Eigen::VectorXd t_scaled;
    Eigen::VectorXd change_points;
    double scale_period;
    Eigen::VectorXd normalized_y;
    double sigma_obs;
    double sigma_k;
    double sigma_m;
    double sigma;
    double tau;
};

lbfgsfloatval_t evaluate(void* instance, const lbfgsfloatval_t* x, lbfgsfloatval_t* g, const int n, const lbfgsfloatval_t step) {
    const Eigen::Map<const Eigen::VectorXd> params_vec(x, n);
    Eigen::Map<Eigen::VectorXd> grad_out(g, n);

    // Extract additional arguments from the instance
    auto* data = static_cast<OptimizationData*>(instance);

    double mlp;
    minus_log_posterior_and_gradient(params_vec,
                                     data->t_scaled,
                                     data->change_points,
                                     data->scale_period,
                                     data->normalized_y,
                                     data->sigma_obs,
                                     data->sigma_k,
                                     data->sigma_m,
                                     data->sigma,
                                     data->tau, mlp, grad_out);

    return mlp;
}

int progress(void* instance, const lbfgsfloatval_t* x, const lbfgsfloatval_t* g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k, int ls) {
    std::cout << "Iteration " << k << ": fx = " << fx << ", xnorm = " << xnorm << ", gnorm = " << gnorm << ", step = " << step << std::endl;
    return 0;
}

extern "C" {
    void optimize(double* params,
                  int params_size,
                  double* t_scaled,
                  int t_scaled_size,
                  double* change_points,
                  int change_points_size,
                  double scale_period,
                  double* normalized_y,
                  int normalized_y_size,
                  double sigma_obs,
                  double sigma_k,
                  double sigma_m,
                  double sigma,
                  double tau) {

        lbfgs_parameter_t param;
        lbfgs_parameter_init(&param);
        param.max_iterations = 10000;
        param.epsilon = 1e-5; // Similar to scipy's `gtol` parameter
        param.past = 5; // Number of past iterations to look back
        param.delta = 1e-9; // Similar to `ftol` in scipy
        param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE; // Strong Wolfe condition
        param.max_linesearch = 20; // Maximum number of line search steps per iteration
        param.min_step = 1e-20;
        param.max_step = 1e20;
        param.ftol = 1e-4; // Line search parameter
        param.wolfe = 0.9; // Wolfe condition parameter
        param.xtol = 1e-16; // Tolerance for machine precision

        lbfgsfloatval_t fx;

        // Use the same instance of data structures within the loop to avoid repeated allocation
        OptimizationData data {
            Eigen::Map<Eigen::VectorXd>(t_scaled, t_scaled_size),
            Eigen::Map<Eigen::VectorXd>(change_points, change_points_size),
            scale_period,
            Eigen::Map<Eigen::VectorXd>(normalized_y, normalized_y_size),
            sigma_obs, sigma_k, sigma_m, sigma, tau
        };

        int ret = lbfgs(params_size, params, &fx, evaluate, progress, &data, &param);

        if (ret == LBFGS_SUCCESS) {
            std::cout << "L-BFGS optimization terminated successfully.\n";
            std::cout << "  fx = " << fx << "\n";
        } else {
            std::cout << "L-BFGS optimization terminated with status code = " << ret << "\n";
        }
    }
}

// To compile, run the following command:
// g++ -std=c++17 -shared -fPIC -Ofast -o liboptimization.so optimize.cpp -llbfgs -I/opt/homebrew/opt/eigen/include/eigen3