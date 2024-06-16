#include <iostream>

#include <ceres/ceres.h>
#include <glog/logging.h> 

const int kNumObservations = 67;

const double data[] = {
  0.000000e+00, 1.133898e+00,
  7.500000e-02, 1.334902e+00,
  1.500000e-01, 1.213546e+00,
  2.250000e-01, 1.252016e+00,
  3.000000e-01, 1.392265e+00,
  3.750000e-01, 1.314458e+00,
  4.500000e-01, 1.472541e+00,
  5.250000e-01, 1.536218e+00,
  6.000000e-01, 1.355679e+00,
  6.750000e-01, 1.463566e+00,
  7.500000e-01, 1.490201e+00,
  8.250000e-01, 1.658699e+00,
  9.000000e-01, 1.067574e+00,
  9.750000e-01, 1.464629e+00,
  1.050000e+00, 1.402653e+00,
  1.125000e+00, 1.713141e+00,
  1.200000e+00, 1.527021e+00,
  1.275000e+00, 1.702632e+00,
  1.350000e+00, 1.423899e+00,
  1.425000e+00, 1.543078e+00,
  1.500000e+00, 1.664015e+00,
  1.575000e+00, 1.732484e+00,
  1.650000e+00, 1.543296e+00,
  1.725000e+00, 1.959523e+00,
  1.800000e+00, 1.685132e+00,
  1.875000e+00, 1.951791e+00,
  1.950000e+00, 2.095346e+00,
  2.025000e+00, 2.361460e+00,
  2.100000e+00, 2.169119e+00,
  2.175000e+00, 2.061745e+00,
  2.250000e+00, 2.178641e+00,
  2.325000e+00, 2.104346e+00,
  2.400000e+00, 2.584470e+00,
  2.475000e+00, 1.914158e+00,
  2.550000e+00, 2.368375e+00,
  2.625000e+00, 2.686125e+00,
  2.700000e+00, 2.712395e+00,
  2.775000e+00, 2.499511e+00,
  2.850000e+00, 2.558897e+00,
  2.925000e+00, 2.309154e+00,
  3.000000e+00, 2.869503e+00,
  3.075000e+00, 3.116645e+00,
  3.150000e+00, 3.094907e+00,
  3.225000e+00, 2.471759e+00,
  3.300000e+00, 3.017131e+00,
  3.375000e+00, 3.232381e+00,
  3.450000e+00, 2.944596e+00,
  3.525000e+00, 3.385343e+00,
  3.600000e+00, 3.199826e+00,
  3.675000e+00, 3.423039e+00,
  3.750000e+00, 3.621552e+00,
  3.825000e+00, 3.559255e+00,
  3.900000e+00, 3.530713e+00,
  3.975000e+00, 3.561766e+00,
  4.050000e+00, 3.544574e+00,
  4.125000e+00, 3.867945e+00,
  4.200000e+00, 4.049776e+00,
  4.275000e+00, 3.885601e+00,
  4.350000e+00, 4.110505e+00,
  4.425000e+00, 4.345320e+00,
  4.500000e+00, 4.161241e+00,
  4.575000e+00, 4.363407e+00,
  4.650000e+00, 4.161576e+00,
  4.725000e+00, 4.619728e+00,
  4.800000e+00, 4.737410e+00,
  4.875000e+00, 4.727863e+00,
  4.950000e+00, 4.669206e+00,
};

// Class definition which computes your residuals/errors. 
// This class will be used with AutoDiffCostFunction
class DataPointResidual {
    // A Functor Class that computes the residual
    public: 
        // The constructor
        // Note that only the constants are passed as arguments to the constructor
        DataPointResidual(double x, double y) : x_(x), y_(y) {}

        // Templated operator()
        template <typename T> 
        bool operator() (const T* const parameters, T* e) const {
            e[0] = y_ - exp( parameters[0]*x_ + parameters[1] ); 
            return true; 
        }

    private:
        double x_, y_; 
};


int main(int argc, char** argv) {

    std::cout << "This program illustrates curve fitting with Ceres solver with an example.\n";

    google::InitGoogleLogging(argv[0]); 

    // Setting the intial values for the parameters that need to be estimated
    // const double initial_m = 0.0;
    // const double initial_c = 0.0;

    const double initial_param[2] = {0.0, 0.0}; 

    double param[2] = {initial_param[0], initial_param[1]}; 
    // double m = initial_m; 
    // double c = initial_c; 

    // Creating the NLLS problem
    ceres::Problem problem;

    // Adding each observation as a residual term in the NLLS objective. 
    // This uses auto-differentiation to obtain the derivative (jacobian).

    // Problem::AddParameterBlock() explicitly adds a parameter block to the Problem. 
    // Optionally it allows the user to associate a Manifold object with the parameter 
    // block too.
    problem.AddParameterBlock(param, 2);

    for (int i = 0; i < kNumObservations; i++) {

        // Creating the residual term as a cost function
        ceres::CostFunction *residual_function = new ceres::AutoDiffCostFunction<DataPointResidual, 1, 2>(
            new DataPointResidual(data[2 * i], data[2 * i + 1]));

        // Adding the residual term to the NLLS problem
        // Note that any variable (in this case m and c) passed to the following AddResidualBlock() function
        // are the variables that will be optimized. 
        problem.AddResidualBlock(residual_function, nullptr, param);
    }

    // Setting solver options
    ceres::Solver::Options options;
    options.max_num_iterations = 100;                // Maximum iterations
    options.linear_solver_type = ceres::DENSE_QR;   // Solver method
    options.minimizer_progress_to_stdout = true;    // If set to true, solver progress is sent to stdout (i.e. printed/logged on terminal)

    // Summary of the various stages of the solver after termination.
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // A brief one line description of the state of the solver after termination.
    std::cout << summary.BriefReport() << "\n";

    // Initial Values of parameters
    std::cout << "Initial m: " << initial_param[0] << " c: " << initial_param[1] << "\n";

    // Final values of parameters after optimization
    std::cout << "Final   m: " << param[0] << " c: " << param[1] << "\n";

    std::cout << "--------------------------------------\n End of Program \n-------------------------------------- \n"; 

    return 0; 

}