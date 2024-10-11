/**
 * \file se3_localization_eskfl.cpp
 *
 *  Created on: Oct 10, 2024
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
 *  The robot pose X is in SE(3) and the landmark positions b_k in R^3,
 *
 *    X = |  R   t  |              // position and orientation
 *        |  0   1  |
 *
 *    b_k = (bx_k, by_k, bz_k)    // lmk coordinates in the world frame
 *
 *  The control signal u is a twist in se(3) comprising longitudinal  velocity
 *  vx and angular velocity wz, with no other velocity components, integrated
 *  over the sampling time dt.
 *
 *    u = (vx*dt, 0, 0, 0, 0, w*dt)
 *
 *  The control is corrupted by additive Gaussian noise u_noise, with covariance
 *
 *    Q = diagonal(sigma_vx^2, sigma_vy^2, sigma_vz^2,
 *                 sigma_wx^2, sigma_wy^2, sigma_wz^2).
 *
 *  This noise accounts for possible lateral and rotational slippage through a
 *  non-zero values of sigma_vy, sigma_vz, sigma_wx and sigma_wy.
 *
 *  At the arrival of a control u, the robot pose is updated with
 *
 *    X_pred = X * Exp(u) = X + u.
 *
 *  Landmark measurements are of the range and bearing type, though they are
 *  put in Cartesian form for simplicity. Their noise n is zero mean Gaussian,
 *  and is specified with a covariances matrix R.
 *
 *  We notice the rigid motion action y = X^-1 * b
 *
 *    y_k = (brx_k, bry_k, brz_k)    // lmk coordinates in the robot frame
 *
 *  We consider the landmark b_k situated at known positions.
 *
 *  We define the pose to estimate as X in SE(3). The estimation error xi
 *  and its covariance P are expressed in the tangent space at X.
 *
 *  All these variables are summarized again as follows
 *
 *    X   : robot pose, SE(3)
 *    u   : robot control, (v*dt; 0; 0; 0; 0; w*dt) in se(3)
 *    Q   : control perturbation covariance
 *    b_k : k-th landmark position, R^3
 *    y   : Cartesian landmark measurement in robot frame, R^3
 *    R   : covariance of the measurement noise
 *
 *  The motion and measurement models are
 *
 *    X_(t+1) = f(X_t, u) = X_t * Exp ( w )     // motion equation
 *    y_k     = h(X, b_k) = X^-1 * b_k          // measurement equation
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
  const int NUMBER_OF_LMKS_TO_MEASURE = 3;
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

  u_nom << 0.1, 0.0, 0.0, 0.0, 0.0, 0.05;
  u_sigmas << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
  Q = (u_sigmas * u_sigmas).matrix().asDiagonal();

  // Declare the Jacobians of the motion wrt robot and control
  manif::SE3d::Jacobian F, W; // F = J_f_x, W = J_f_epsilon;

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

  // Declare the Jacobian of the measurements
  Matrix<double, 3, 6> H; // H = J_h_x
  Matrix3d V; // V = J_h_delta

  // Declare some temporaries
  Vector3d z;  // innovation
  Matrix3d S;  // covariances of the above
  Matrix<double, 6, 3> K;  // Kalman gain
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

  // Make 20 steps. Measure up to three landmarks each time.
  for (int t = 0; t < 20; ++t) {
    //// I. Simulation #########################################################

    /// simulate noise
    u_noise = u_sigmas * Array6d::Random(); // control noise
    u_noisy = u_nom + u_noise;  // noisy control

    u_simu = u_nom;
    u_est = u_noisy;
    u_unfilt = u_noisy;

    /// first we move - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    X_simulation = X_simulation + u_simu; // overloaded X.rplus(u) = X * exp(u)

    /// then we measure all landmarks - - - - - - - - - - - - - - - - - - - -
    for (std::size_t i = 0; i < landmarks.size(); ++i) {
      b = landmarks[i]; // lmk coordinates in world frame

      /// simulate noise
      y_noise = y_sigmas * Array3d::Random(); // measurement noise

      y = X_simulation.inverse().act(b);  // landmark measurement, without noise
      y = y + y_noise;  // landmark measurement, noisy
      measurements[i] = y;  // store for the estimator just below
    }




    //// II. Estimation ###############################################################################
    /// First we move - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    X = X + u_est;  // X * exp(u). We use right plus here, the Jacobians need to be calculated manually
    F.setIdentity();  // F = J_f_x
    W = X.adj();  // W = J_f_epsilon

    P = F * P * F.transpose() + W * Q * W.transpose();


    /// Then we correct using the measurements of each lmk - - - - - - - - -
    for (int i = 0; i < NUMBER_OF_LMKS_TO_MEASURE; ++i) {
      // landmark
      b = landmarks[i]; // lmk coordinates in world frame

      // measurement
      y = measurements[i];  // lmk measurement, noisy

      // innovation
      z = X.act(y) - b; // z = X * y - X * ybar (the second term = b)

      // Jacobians
      H.topLeftCorner<3, 3>() = -Matrix3d::Identity();
      H.topRightCorner<3, 3>() = manif::skew(b);
      V = X.rotation();

      // innovation covariance
      S = H * P * H.transpose() + V * R * V.transpose();

      // Kalman gain
      K = P * H.transpose() * S.inverse();

      // Correction step
      dx = K * z; // dx is in the tangent space at X

      // Update
      X = X.lplus(dx);  // overloaded X.lplus(dx) = exp(dx) * X
      P = (I - K * H) * P * (I - K * H).transpose()
           + K * V * R * V.transpose() * K.transpose();
    }




    //// III. Unfiltered ##############################################################################

    // move also an unfiltered version for comparison purposes
    X_unfiltered = X_unfiltered + u_unfilt;




    //// IV. Results ##############################################################################

    // DEBUG
    cout << "X simulated : " << X_simulation.log().coeffs().transpose() << endl;
    cout << "X estimated : " << X.log().coeffs().transpose() << endl;
    cout << "X unfilterd : " << X_unfiltered.log().coeffs().transpose() << endl;
    cout << "-------------------------------------------------------" << endl;
    // END DEBUG

  }

  //
  //
  // END OF TEMPORAL LOOP. DONE.

  return 0;
}
