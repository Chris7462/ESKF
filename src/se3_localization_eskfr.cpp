/**
 * \file se3_localization_eskfr.cpp
 *
 *  Created on: October 04, 2024
 *    \author: Yi-Chen Zhang
 *
 *  ---------------------------------------------------------------------------
 *  Demonstration example:
 *
 *  3D Robot localization based on position measurements (GPS-like)
 *  using the (Left) Invariant Extended Kalman Filter method.
 *
 *  The following example corresponds to the simulation section in the ESKF on
 *  Matrix Lie Groups paper. Please refer to the paper for further details.
 *  ---------------------------------------------------------------------------
 *
 *  We consider a robot in 3D space. The robot receives control actions in the
 *  form of axial and angular velocities, and is able to measure its position
 *  using a GPS for instance.
 *
 *  The robot pose X is in SE(3) and the GPS measurement y_k is in R^3,
 *
 *    X = |  R  t  |              // position and orientation
 *        |  0  1  |
 *
 *  The control signal u is a twist in se(3) comprising longitudinal velocity v
 *  and angular velocity w, with no lateral velocity component, integrated over
 *  the sampling time dt.
 *
 *    u = (v*dt, 0, 0, 0, 0, w*dt)
 *
 *  The control is corrupted by additive Gaussian noise u_noise, with covariance
 *
 *    Q = diagonal(sigma_vx^2, sigma_vy^2, sigma_vz^2,
 *                 sigma_wx^2, sigma_wy^2, sigma_wz^2).
 *
 *  This noise accounts for possible lateral and rotational slippage through a
 *  non-zero values of sigma_vy, sigma_vz, sigma_wx and sigma_wy.
 *
 *  At the arrival of a control u, the robot pose is updated
 *
 *    X_pred = X * Exp(u) = X + u.
 *
 *  GPS measurements are put in Cartesian form for simplicity. Their noise n is
 *  zero mean Gaussian, and is specified with a covariances matrix R.
 *
 *  We notice the rigid motion action y = h(X) = X * [0 0 0]^T + delta
 *
 *      y_k = (x, y, z)       // robot coordinates
 *
 *  We define the pose to estimate as X in SE(3). The estimation error dx and
 *  its covariance P are expressed in the global space at epsilon.
 *
 *  All these variables are summarized again as follows
 *
 *    X : robot pose, SE(3)
 *    u : robot control, (v*dt; 0; 0; 0; 0; w*dt) in se(3)
 *    Q : control perturbation covariance
 *    y : robot position measurement in global frame, R^3
 *    R : covariance of the measurement noise
 *
 *  The motion and measurement models are
 *
 *    X_(t+1) = f(X_t, u) = X_t * Exp ( u )     // motion equation
 *    y_k     = h(X, b_k) = X * [0 0 1]^T       // measurement equation
 *
 *  The algorithm below comprises first a simulator to produce measurements,
 *  then uses these measurements to estimate the state, using the ESKF on
 *  matrix Lie group.
 *
 *  Printing simulated state and estimated state together with an unfiltered
 *  state (i.e. without Kalman corrections) allows for evaluating the quality
 *  of the estimates.
 */


#include <manif/SE3.h>

#include <vector>
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;

using namespace Eigen;

using Array6d = Array<double, 6, 1>;
using Vector6d = Matrix<double, 6, 1>;
using Matrix6d = Matrix<double, 6, 6>;

int main()
{
  std::srand((unsigned int) time(0));

  // START CONFIGURATION
  //
  //
  const Matrix6d I = Matrix6d::Identity();

  // Define the robot pose element and its covariance
  manif::SE3d X, X_simulation, X_unfiltered;
  Matrix6d P;

  X_simulation.setIdentity();
  X.setIdentity();
  X_unfiltered.setIdentity();
  P.setZero();

  // Define a control vector and its noise and covariance
  manif::SE3Tangentd u_simu, u_est, u_unfilt;
  Vector6d u_nom, u_noisy, u_noise;
  Array6d u_sigmas;
  Matrix6d Q;

  u_nom << 0.1, 0.0, 0.0, 0.0, 0.0, 0.1;
  u_sigmas << 0.1, 0.1, 0.1, 0.05, 0.05, 0.05;
  Q = (u_sigmas * u_sigmas).matrix().asDiagonal();

  // Declare the Jacobians of the motion wrt robot and control
  manif::SE3d::Jacobian F, W; // F = J_f_x and W = J_f_epsilon;

  // Define the gps measurements in R^3
  Vector3d y, y_noise;
  Array3d y_sigmas;
  Matrix3d R;

  y_sigmas << 0.01, 0.01, 0.01;
  R = (y_sigmas * y_sigmas).matrix().asDiagonal();

  // Declare the Jacobian of the measurements wrt the robot pose
  Matrix<double, 3, 6> H; // H = J_h_x
  Matrix3d V; // V = J_h_delta

  // Declare some temporaries
  Vector3d z; // innovation
  Matrix3d S; // covariances of the above
  Matrix<double, 6, 3> K; // Kalman gain
  manif::SE3Tangentd dx;  // optimal update step, or error-state

  //
  //
  // CONFIGURATION DONE



  // DEBUG
  cout << std::fixed   << std::setprecision(3) << std::showpos << endl;
  cout << "X STATE     :    X      Y      Z    TH_x   TH_y   TH_z " << endl;
  cout << "-------------------------------------------------------" << endl;
  cout << "X initial   : " << X_simulation.log().coeffs().transpose() << endl;
  cout << "-------------------------------------------------------" << endl;
  // END DEBUG




  // START TEMPORAL LOOP
  //
  //

  // Make 1000 steps. Measure one GPS position each time.
  for (int t = 0; t < 1000; ++t) {
    //// I. Simulation #########################################################

    /// simulate noise
    u_noise = u_sigmas * Array6d::Random(); // control noise
    u_noisy = u_nom + u_noise;  // noisy control

    u_simu = u_nom;
    u_est = u_noisy;
    u_unfilt = u_noisy;

    /// first we move - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    X_simulation = X_simulation + u_simu; // overloaded X.rplus(u) = X * exp(u)

    /// then we receive noisy gps measurement - - - - - - - - - - - - - - - -
    y_noise = y_sigmas * Array3d::Random(); // simulate measurement noise

    y = X_simulation.translation(); // position measurement, before adding noise
    y = y + y_noise;  // position measurement, noisy




    //// II. Estimation ###############################################################################

    /// First we move - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    X = X.plus(u_est, F, W);  // X * exp(u), with Jacobians

    P = F * P * F.transpose() + W * Q * W.transpose();


    /// Then we correct using the gps position - - - - - - - - - - - - - - -

    // innovation
    z = X.inverse().act(y); // z = X^{-1} * y - X^{-1} * ybar (the second term is 0).

    // Jacobians
    H.topLeftCorner<3, 3>() = Matrix3d::Identity();
    H.topRightCorner<3, 1>() = Vector3d::Zero();
    V = X.inverse().rotation();

    // innovation covariance
    S = H * P * H.transpose() + V * R * V.transpose();

    // Kalman gain
    K = P * H.transpose() * S.inverse();

    // Correction step
    dx = K * z;

    // Update
    X = X.plus(dx);
    P = (I - K * H) * P * (I - K * H).transpose()
        + K * V * R * V.transpose() * K.transpose();




    //// III. Unfiltered ##############################################################################

    // move also an unfiltered version for comparison purposes
    X_unfiltered = X_unfiltered + u_unfilt;




    //// IV. Results ##############################################################################

    // DEBUG
    cout << "X simulated : " << X_simulation.log().coeffs().transpose() << endl;
    cout << "X estimated : " << X.log().coeffs().transpose() << endl;
    cout << "X unfilterd : " << X_unfiltered.log().coeffs().transpose() << endl;
    cout << "----------------------------------" << endl;
    // END DEBUG

  }

  //
  //
  // END OF TEMPORAL LOOP. DONE.

  return 0;
}
