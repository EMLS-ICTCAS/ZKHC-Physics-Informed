#ifndef MISC_HPP
#define MISC_HPP

// auxiliary macros to access the fields
#define U(i, j) u[(i) * prm.NY + (j)]
#define V(i, j) v[(i) * prm.NY + (j)]
#define W(i, j) w[(i) * prm.NY + (j)]
#define ADV_U(i, j) adv_u[(i) * prm.NY + (j)]
#define ADV_V(i, j) adv_v[(i) * prm.NY + (j)]
#define Ustar(i, j) ustar[(i) * prm.NY + (j)]
#define Vstar(i, j) vstar[(i) * prm.NY + (j)]
#define P(i, j) p[(i) * prm.NY + (j)]
#define DIV(i, j) div((i) * prm.ny + (j))
#define x(i) (-0.5 * prm.dx + (i) * prm.dx)
#define y(j) (-0.5 * prm.dy + (j) * prm.dy)

typedef struct prm {
  int NX;    // number of points in the x direction (including the ghost points)
  int NY;    // number of points in the y direction (including the ghost points)
  int nx;    // number of points in the x direction (excluding the ghost points)
  int ny;    // number of points in the y direction (excluding the ghost points)
  double LX; // length of the domain in the x direction
  double LY; // length of the domain in the y direction
  double dt; // time step
  double dx; // grid spacing in the x direction
  double dy; // grid spacing in the y direction
  double T;  // final time
  double U;  // characteristic velocity
  double L;  // characteristic length
  double nu; // kinematic viscosity
  double Re; // Reynolds number
  size_t
      NXNY; // total number of points in the domain (including the ghost points)
  size_t
      nxny; // total number of points in the domain (excluding the ghost points)
  bool obstacle_ON; // true if there is an obstacle in the domain, false
                    // otherwise
} Prm;

#endif // MISC_HPP
