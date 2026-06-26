#ifndef NS_HPP
#define NS_HPP

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include "../include/object.hpp"
#include "misc.hpp"

// auxiliary macros to access the fields
#define p(jj) (obstacle.InterpolatingPoints[6 * k + (jj)])
#define m_x (obstacle.MirrorPoints[k].x)
#define m_y (obstacle.MirrorPoints[k].y)

typedef Eigen::SparseMatrix<double>
    SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> Trip; // declares a triplet type of double

using namespace std;

// @brief 1st order semi-Lagrangian for advection
// @param u velocity in x direction
// @param v velocity in y direction
// @param q scalar field to be advected
// @param prm parameters of the simulation (dx, dy, dt, etc.)
// @param sign sign of the direction of the velocity fields (to be easily
// inverted). If unsure, use 1.
// @param obstacle object representing the obstacle in the domain
void Semilag(double *u, double *v, double *q, const Prm &prm, int sign,
             Object &obstacle);

// @brief 2nd order semi-Lagrangian for advection
// @param u velocity in x direction
// @param v velocity in y direction
// @param q0 initial scalar field to be advected
// @param q1 final scalar field to be advected
// @param prm parameters of the simulation (dx, dy, dt, etc.)
// @param obstacle object representing the obstacle in the domain
void Semilag2(double *u, double *v, double *q0, double *q1, const Prm &prm,
              Object &obstacle);

// @brief Set the boundary conditions for the velocity fields, but on the ghost
// points (i.e. fictitious points outside the domain) by linear interpolation
// @param u velocity in x direction
// @param v velocity in y direction
// @param prm parameters of the simulation (dx, dy, dt, etc.)
// @param obstacle object representing the obstacle in the domain
void BC_velocity(double *u, double *v, const Prm &prm);

// @brief Set the boundary conditions for the pressure field, but on the ghost
// points (i.e. fictitious points outside the domain) by linear interpolation
// @param p pressure
// @param prm parameters of the simulation (dx, dy, dt, etc.)
// @param obstacle object representing the obstacle in the domain
void BC_pressure(double *p, const Prm &prm);

// @brief Interpolate the field phi at a given point of the grid (i, j) based on
// the values of the field at the surrounding points. In practice it is used on
// the points just outside the obstacle to decrease the gradient of the field
// between the fluid and the obstacle
// @param i x index of the point
// @param j y index of the point
// @param phi field to be interpolated
// @param prm parameters of the simulation (dx, dy, dt, etc.)
// @param obstacle object representing the obstacle in the domain
// @return the interpolated value of the field at the point (i, j)
double interpolate(int i, int j, double *phi, const Prm &prm, Object &obstacle);

// @brief Set the vorticity field based on the velocity fields
// @param u velocity in x direction
// @param v velocity in y direction
// @param w vorticity
// @param prm parameters of the simulation (dx, dy, dt, etc.)
void set_vorticity(double *u, double *v, double *w, const Prm &prm);

// @brief Build the Poisson matrix for the pressure Poisson equation poisson(p)
// = div(u)
// @param coeffs vector of triplets to store the coefficients of the Possion
// matrix. Triplets consist of (i, j, value), where i and j are the indices of
// the matrix and value is the value of the coefficient
// @param prm parameters of the simulation (dx, dy, dt, etc.)
void buildPoissonMatrix(vector<Trip> &coeffs, const Prm &prm);

#endif // NS_HPP
