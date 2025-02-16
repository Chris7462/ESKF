/**
 * \file se_2_3_localization_eskfl.cpp
 *
 *  Created on: October 28, 2024
 *    \author: Yi-Chen Zhang
 *
 *  ----------------------------------------------------------------------------
 *  Demonstration example:
 *
 *  3D Robot localization based on fixed landmarks using the ESKF on the Matrix
 *  Lie Group of the left form, which is equivalent to LIEKF.
 *
 *  The following example corresponds to the simulation section in the ESKF on
 *  Matrix Lie Groups paper. Please refer to the paper for further details.
 *  ----------------------------------------------------------------------------
 *
 *  We consider a robot in 3D space surrounded by a small
 *  number of landmarks. The robot receives control actions in the form of axial
 *  and angular velocities, and is able to measure the location of the beacons
 *  with respect to its own reference frame.
 *
 *  We assume in this example that the IMU frame coincides with the robot frame.
 *
 *  The robot extended pose X is in SE_2(3) and the landmark position b_k in R^3,
 *
 *    X = | R  v  p |   // orientation, linear velocity, and position
 *        |    1    |
 *        |       1 |
 *
 *    b_k = (bx_k, by_k, bz_k)    // lmk coordinates in the world frame
 *
 *    alpha_k = (alphax_k, alphay_k, alphaz_k)  // linear accelerometer measurements in IMU frame
 *
 *    omega_k = (omegax_k, omegay_k, omegaz_k)  // gyroscope measurements in IMU frame
 *
 *    g = (0, 0, -9.80665)  // acceleration due to gravity in world frame
 *
 * Consider robot coordinate frame b and world coordinate frame w.
 * - p is the position of the origin of the robot frame b with respect to the world frame w
 * - R is the orientation of the robot frame b with respect to the world frame w
 * - v is the velocity of the origin of the robot frame b with respect to the world frame w
 *
 * The continuous-time system dynamics without the noise can be represented as
 * d/dt R_{t} = R_{t}(\tilde{\omega}_t - \epsilon_{\omega, t})^{\wedge}
 * d/dt v_{t} = R_{t}(\tilde{\alpha}_{t} - \epsilon_{\alpha, t}) + g
 * d/dt p_{t} = v_{t}
 *
 * The discrete-time kinematic model from step k to k+1 can be written as,
 * R_{k+1} = R_{k}\Gamma_{0}(\omega_{k}\Delta t)
 * v_{k+1} = v_{k} + R_{k}\Gamma_{1}(\omega_{k}\Delta t)\alpha_{k}\Delta t + g\Delta t
 * p_{k+1} = p_{k} + v_{k}\Delta t + R_{k}\Gamma_{2}(\omega_{k}\Delta t)\alpha_{k}\Delta t^{2} + \frac{1}{2}g\Delta t^{2}
 *
 * The exponential mapping of SE_2(3) is defined as,
 * for u = [u_p, u_w, u_v]^{T}
 * Exp(u) = | Exp_SO3(u_w)   JlSO3(u_w) u_v   JlSO3(u_w) u_p |
 *          | 0    0    0                 1                0 |
 *          | 0    0    0                 0                1 |
 * where, JlSO3 is the left Jacobian of the SO(3) group.
 *
 */


#include <vector>
#include <iostream>
#include <iomanip>
#include <ctime>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

using std::cout;
using std::endl;

using namespace Eigen;

using Matrix5d = Matrix<double, 5, 5>;

using Array9d = Array<double, 9, 1>;
using Vector9d = Matrix<double, 9, 1>;
using Matrix9d = Matrix<double, 9, 9>;

Matrix3d skew(const Vector3d & u);

Matrix3d gamma0(const Vector3d & phi);

Matrix3d gamma1(const Vector3d & phi);

Matrix3d gamma2(const Vector3d & phi);

Matrix5d makeTwist(const Vector9d & u);

Matrix9d adj(const Matrix5d & X);

int main()
{
  std::srand((unsigned int) time(0));

  // START CONFIGURATION
  //
  //
  const int NUMBER_OF_LMKS_TO_MEASURE = 3;
  const Matrix9d I = Matrix9d::Identity();

  // Define the robot extended pose element and its covariance
  Matrix5d X, X_simulation, X_unfiltered;
  Matrix9d P;

  X_simulation.setIdentity();
  X.setIdentity();
  X_unfiltered.setIdentity();
  P.setZero();

  // acceleration due to gravity in world frame
  Vector3d g;
  g << 0, 0, -9.80665;
  const double dt = 0.01;

  // IMU measurements in IMU frame
  Vector3d alpha, omega, const_alpha;
  const_alpha << 0.2, 0.0, 0.0; // constant acceleration in IMU frame without gravity compensation
  alpha = const_alpha - X_simulation.block<3, 3>(0, 0).transpose() * g;
  omega << 0.0, 0.0, 0.1; // constant angular velocity about z-direction (yaw rate) in IMU frame

  // Previous IMU measurements in IMU frame initialized to values expected when stationary
  Vector3d alpha_prev, omega_prev;
  alpha_prev = alpha;
  omega_prev.setZero();

  // Define a control vector and its noise and covariance
  Array9d u_sigmas;
  Matrix9d Q;

  u_sigmas << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0;
           // roll, pitch, yaw, vx, vy, vz, x, y, z
  Q = (u_sigmas * u_sigmas).matrix().asDiagonal();

  // Declare the Jacobians of the motion wrt robot and control
  Matrix9d F, W;  // F = J_f_x and W = J_f_epsilon;

  // Define five landmarks in R^3
  Vector3d b0, b1, b2, b3, b4, b;
  b0 << 2.0, 0.0, 0.0;
  b1 << 3.0, -1.0, -1.0;
  b2 << 2.0, -1.0, 1.0;
  b3 << 2.0, 1.0, 1.0;
  b4 << 2.0, 1.0, -1.0;

  std::vector<Vector3d> landmarks {b0, b1, b2, b3, b4};

  // Define the landmarks' measurements
  Vector3d y, y_noise;
  Array3d y_sigmas;
  Matrix3d R;
  std::vector<Vector3d> measurements(landmarks.size());

  y_sigmas << 0.01, 0.01, 0.01;
  R = (y_sigmas * y_sigmas).matrix().asDiagonal();

  // Declare the Jacobian of the measurements wrt the robot pose
  Matrix<double, 3, 9> H; // H = J_e_x
  Matrix3d V; // V = J_h_delta

  // Declare some temporaries
  Vector3d z; // innovation
  Matrix3d S; // covariances of the above
  Matrix<double, 9, 3> K; // Kalman gain

  //
  //
  // CONFIGURATION DONE



  // DEBUG
  cout << std::fixed   << std::setprecision(3) << std::showpos << endl;
  cout << "X STATE     :   TH_x   TH_y   TH_z    V_x    V_y    V_z      X      Y      Z" << endl;
  cout << "----------------------------------------------------------------------------" << endl;
  cout << "X simulated : " << X_simulation.block<3, 3>(0, 0).eulerAngles(2, 1, 0).reverse().transpose()
                           << " " << X_simulation.block<3, 1>(0, 3).transpose()
                           << " " << X_simulation.block<3, 1>(0, 4).transpose() << endl;
  cout << "X estimated : " << X.block<3, 3>(0, 0).eulerAngles(2, 1, 0).reverse().transpose()
                           << " " << X.block<3, 1>(0, 3).transpose()
                           << " " << X.block<3, 1>(0, 4).transpose() << endl;
  cout << "X unfilterd : " << X_unfiltered.block<3, 3>(0, 0).eulerAngles(2, 1, 0).reverse().transpose()
                           << " " << X_unfiltered.block<3, 1>(0, 3).transpose()
                           << " " << X_unfiltered.block<3, 1>(0, 4).transpose() << endl;
  cout << "----------------------------------------------------------------------------" << endl;
  // END DEBUG


  // START TEMPORAL LOOP
  //
  //

  // Make 20 steps. Measure up to three landmarks each time.
  for (double t = 0; t < 100; t += dt) {
    //// I. Simulation ###############################################################################

    /// get current simulated state and measurements from previous step
    Matrix3d R_k = X_simulation.block<3, 3>(0, 0);
    Vector3d v_k = X_simulation.block<3, 1>(0, 3);
    Vector3d p_k = X_simulation.block<3, 1>(0, 4);

    /// first we move - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    X_simulation.block<3, 3>(0, 0) = R_k * gamma0(omega_prev * dt);
    X_simulation.block<3, 1>(0, 3) = v_k + R_k * gamma1(omega_prev * dt) * alpha_prev * dt + g * dt;
    X_simulation.block<3, 1>(0, 4) = p_k + v_k * dt + R_k * gamma2(omega_prev * dt) * alpha_prev * dt * dt  + 0.5 * g * dt * dt;

    /// update expected IMU measurements
    alpha = const_alpha - X_simulation.block<3, 3>(0, 0).transpose() * g; // update expected IMU measurement after moving

    /// then we measure all landmarks - - - - - - - - - - - - - - - - - - - -
    for (std::size_t i = 0; i < landmarks.size(); ++i) {
      b = landmarks[i]; // lmk coordinates in world frame

      /// simulate noise
      y_noise = y_sigmas * Array3d::Random(); // measurement noise

      auto X_sim_inv = X_simulation.inverse();
      y = X_sim_inv.block<3, 3>(0, 0) * b + X_sim_inv.block<3, 1>(0, 4);  // landmark measurement, without noise
      y = y + y_noise;  // landmark measurement, noisy
      measurements[i] = y;  // store for the estimator just below
    }




    //// II. Estimation ###############################################################################

    /// get current state estimate to build the state-dependent control vector
    Matrix3d R_k_est = X.block<3, 3>(0, 0);
    Vector3d v_k_est = X.block<3, 1>(0, 3);
    Vector3d p_k_est = X.block<3, 1>(0, 4);

    /// simulate noise
    Vector3d alpha_noisy = alpha_prev + 0.05 * Vector3d::Random();
    Vector3d omega_noisy = omega_prev + 0.01 * Vector3d::Random();

    /// First we move - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    X.block<3, 3>(0, 0) = R_k_est * gamma0(omega_noisy * dt);
    X.block<3, 1>(0, 3) = v_k_est + R_k_est * gamma1(omega_noisy * dt) * alpha_noisy * dt + g * dt;
    X.block<3, 1>(0, 4) = p_k_est + v_k_est * dt + R_k_est * gamma2(omega_noisy * dt) * alpha_noisy * dt * dt  + 0.5 * g * dt * dt;

    // Prepare Jacobian of state-dependent control vector
    Matrix9d A = Matrix9d::Zero();
    A.block<3, 3>(3, 0) = skew(g);
    A.block<3, 3>(6, 3) = Matrix3d::Identity();

    // Left invariant Jacobians
    auto & F = (A * dt).exp();
    W = adj(X);

    P = F * P * F.transpose() + W * Q * W.transpose();


    /// Then we correct using the measurements of each lmk - - - - - - - - -
    for (int i = 0; i < NUMBER_OF_LMKS_TO_MEASURE; ++i) {
      // landmark
      b = landmarks[i]; // lmk coordinates in world frame

      // measurement
      y = measurements[i];  // lmk measurement, noisy

      // innovation
      z = X.block<3, 3>(0, 0) * y + X.block<3, 1>(0, 4) - b; // z = X * y - X * ybar (the second term = b)

      // Jacobians
      H.setZero();
      H.topLeftCorner<3, 3>() = skew(b);
      H.topRightCorner<3, 3>() = -Matrix3d::Identity();
      V = X.block<3, 3>(0, 0);

      // innovation covariance
      S = H * P * H.transpose() + V * R * V.transpose();

      // Kalman gain
      K = P * H.transpose() * S.inverse();

      // Correction step
      Vector9d dx = K * z;
      Matrix5d xi = makeTwist(dx);

      // Update
      X = xi.exp() * X; // left plus
      P = (I - K * H) * P * (I - K * H).transpose()
          + K * V * R * V.transpose() * K.transpose();
    }




    //// III. Unfiltered ##############################################################################

    // move also an unfiltered version for comparison purposes
    Matrix3d R_k_unfiltered = X_unfiltered.block<3, 3>(0, 0);
    Vector3d v_k_unfiltered = X_unfiltered.block<3, 1>(0, 3);
    Vector3d p_k_unfiltered = X_unfiltered.block<3, 1>(0, 4);

    X_unfiltered.block<3, 3>(0, 0) = R_k_unfiltered * gamma0(omega_noisy * dt);
    X_unfiltered.block<3, 1>(0, 3) = v_k_unfiltered + R_k_unfiltered * gamma1(omega_noisy * dt) * alpha_noisy * dt + g * dt;
    X_unfiltered.block<3, 1>(0, 4) = p_k_unfiltered + v_k_unfiltered * dt + R_k_unfiltered * gamma2(omega_noisy * dt) * alpha_noisy * dt * dt  + 0.5 * g * dt * dt;


    alpha_prev = alpha;
    omega_prev = omega;


    //// IV. Results ##############################################################################

    // DEBUG
    cout << "X simulated : " << X_simulation.block<3, 3>(0, 0).eulerAngles(2, 1, 0).reverse().transpose()
                             << " " << X_simulation.block<3, 1>(0, 3).transpose()
                             << " " << X_simulation.block<3, 1>(0, 4).transpose() << endl;
    cout << "X estimated : " << X.block<3, 3>(0, 0).eulerAngles(2, 1, 0).reverse().transpose()
                             << " " << X.block<3, 1>(0, 3).transpose()
                             << " " << X.block<3, 1>(0, 4).transpose() << endl;
    cout << "X unfilterd : " << X_unfiltered.block<3, 3>(0, 0).eulerAngles(2, 1, 0).reverse().transpose()
                             << " " << X_unfiltered.block<3, 1>(0, 3).transpose()
                             << " " << X_unfiltered.block<3, 1>(0, 4).transpose() << endl;
    cout << "---------------------------------------------------------------------------" << endl;

//    cout << "X simulated : log: " << X_simulation.log() << endl;
//    cout << "X estimated : log: " << X.log() << endl;
//    cout << "X unfilterd : log: " << X_unfiltered.log() << endl;
//    cout << "---------------------------------------------------------------------------" << endl;
//    cout << "---------------------------------------------------------------------------" << endl;
//    // END DEBUG

  }

  //
  //
  // END OF TEMPORAL LOOP. DONE.

  return 0;
}

Matrix3d skew(const Vector3d & u)
{
  return (Matrix3d() <<
    0.0, -u(2), u(1),
    u(2), 0.0, -u(0),
    -u(1), u(0), 0.0).finished();
}

Matrix3d gamma0(const Vector3d & phi)
{
  const double norm_phi = phi.norm();
  if (norm_phi > 1e-8) {
    auto skew_phi = skew(phi);
    return (Matrix3d::Identity() + sin(norm_phi) / norm_phi * skew_phi +
      (1 - cos(norm_phi)) / pow(norm_phi, 2) * skew_phi * skew_phi);
  }
  return Matrix3d::Identity();
}

Matrix3d gamma1(const Vector3d & phi)
{
  const double norm_phi = phi.norm();
  if (norm_phi > 1e-8) {
    auto skew_phi = skew(phi);
    return (Matrix3d::Identity() + (1 - cos(norm_phi)) / pow(norm_phi, 2) * skew_phi +
      (norm_phi - sin(norm_phi)) / pow(norm_phi, 3) * skew_phi * skew_phi);
  }
  return Matrix3d::Identity();
}

Matrix3d gamma2(const Vector3d & phi)
{
  const double norm_phi = phi.norm();
  if (norm_phi > 1e-8) {
    auto skew_phi = skew(phi);
    return (0.5 * Matrix3d::Identity() +
      (norm_phi - sin(norm_phi)) / pow(norm_phi, 3) * skew_phi +
      (pow(norm_phi, 2) + 2 * cos(norm_phi) - 2) / (2 * pow(norm_phi, 4)) * skew_phi * skew_phi);
  }
  return 0.5 * Matrix3d::Identity();
}

Matrix5d makeTwist(const Vector9d & u)
{
  Matrix5d twist = Matrix5d::Zero();
  twist.block<3, 3>(0, 0) = skew(u.block<3, 1>(0, 0));
  twist.block<3, 1>(0, 3) = u.block<3, 1>(3, 0);
  twist.block<3, 1>(0, 4) = u.block<3, 1>(6, 0);
  return twist;
}

Matrix9d adj(const Matrix5d & X)
{
  Matrix9d adj = Matrix9d::Zero();
  adj.block<3, 3>(0, 0) = X.block<3, 3>(0, 0);
  adj.block<3, 3>(3, 0) = skew(X.block<3, 1>(0, 3)) * X.block<3, 3>(0, 0);
  adj.block<3, 3>(6, 0) = skew(X.block<3, 1>(0, 4)) * X.block<3, 3>(0, 0);
  adj.block<3, 3>(3, 3) = X.block<3, 3>(0, 0);
  adj.block<3, 3>(6, 6) = X.block<3, 3>(0, 0);

  return adj;
}
