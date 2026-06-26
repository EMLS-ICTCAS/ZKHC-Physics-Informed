#include "../include/NS.hpp"

#include <cstring>

void Semilag(double *u, double *v, double *q, const Prm &prm, int sign,
             Object &obstacle) {
  int sign_u, sign_v;
  double a, b;
  double *aux = (double *)calloc(prm.NXNY, sizeof(double));
  for (int i = 1; i < prm.NX - 1; i++) {
    for (int j = 1; j < prm.NY - 1; j++) {
      if (prm.obstacle_ON && obstacle.IsInside[i * prm.NY + j]) {
        continue;
      }
      if (sign * U(i, j) > 0) {
        a = 1 - sign * U(i, j) * prm.dt / prm.dx;
        sign_u = 1;
      } else {
        a = 1 + sign * U(i, j) * prm.dt / prm.dx;
        sign_u = -1;
      }
      if (sign * V(i, j) > 0) {
        b = 1 - sign * V(i, j) * prm.dt / prm.dy;
        sign_v = 1;
      } else {
        b = 1 + sign * V(i, j) * prm.dt / prm.dy;
        sign_v = -1;
      }

      aux[i * prm.NY + j] =
          a * b * q[i * prm.NY + j] +
          (1 - a) * b * q[(i - sign_u) * prm.NY + j] +
          a * (1 - b) * q[i * prm.NY + j - sign_v] +
          (1 - a) * (1 - b) * q[(i - sign_u) * prm.NY + j - sign_v];
    }
  }
  memcpy(q, aux, prm.NXNY * sizeof(double));
  free(aux);
}

void Semilag2(double *u, double *v, double *q0, double *q1, const Prm &prm,
              Object &obstacle) {
  memcpy(q1, q0, prm.NXNY * sizeof(double));
  Semilag(u, v, q1, prm, 1, obstacle);
  Semilag(u, v, q1, prm, -1, obstacle);
  for (int i = 0; i < prm.NX * prm.NY; i++)
    q1[i] = q0[i] + (q0[i] - q1[i]) / 2;
  Semilag(u, v, q1, prm, 1, obstacle);
}

void BC_velocity(double *u, double *v, const Prm &prm) {
  for (int i = 0; i < prm.NX; i++) {
    // bottom boundary: \partial_y u = 0, v = 0
    U(i, 0) = U(i, 1);
    V(i, 0) = -V(i, 1);
    // top boundary: \partial_y u = 0, v = 0
    U(i, prm.NY - 1) = U(i, prm.NY - 2);
    V(i, prm.NY - 1) = -V(i, prm.NY - 2);
  }

  for (int j = 0; j < prm.NY; j++) {
    // left boundary: u = 1, v = 0
    U(0, j) = 2 - U(1, j);
    V(0, j) = -V(1, j);
    // right boundary: \partial_x u = 0, \partial_x v = 0
    U(prm.NX - 1, j) = U(prm.NX - 2, j);
    V(prm.NX - 1, j) = V(prm.NX - 2, j);
  }
}

void BC_pressure(double *p, const Prm &prm) {
  for (int i = 0; i < prm.NX; i++) {
    // bottom boundary: \partial_y p = 0
    P(i, 0) = P(i, 1);
    // top boundary: \partial_y p = 0
    P(i, prm.NY - 1) = P(i, prm.NY - 2);
  }
  for (int j = 0; j < prm.NY; j++) {
    // left boundary: \partial_x p = 0
    P(0, j) = P(1, j);
    // right boundary: p = 0
    P(prm.NX - 1, j) = -P(prm.NX - 2, j); // mine
  }
}

double interpolate(int i, int j, double *phi, const Prm &prm,
                   Object &obstacle) {
  // interpolate with the 8 surrounding points that are not forcing points (x is
  // the point (i, j) and o are the surrounding points, some of them may be
  // forcing points) o o o o x o o o o
  int I = i - 1, J = j - 1;
  double x = x(i), y = y(j);
  double interp = 0;
  double dx, dy, tmp, weight, total_weight = 0;
  for (int k = 0; k < 3; k++) {
    for (int m = 0; m < 3; m++) {
      if (obstacle.IsInterface[(I + k) * prm.NY + J + m])
        continue;
      if (obstacle.IsGhost[(I + k) * prm.NY + J + m]) {
        // in this case we use the boundary value
        dx = x - obstacle.BoundaryPoints[(I + k) * prm.NY + J + m].x;
        dy = y - obstacle.BoundaryPoints[(I + k) * prm.NY + J + m].y;
      } else {
        dx = x - x(I + k);
        dy = y - y(J + m);
      }
      tmp = dx * dx + dy * dy;
      if (tmp < 1e-8) // the point is in the boundary
        return 0;
      weight = 1.0 / tmp;
      interp += weight * phi[(I + k) * prm.NY + J + m];
      total_weight += weight;
    }
  }
  return interp / total_weight;
}

void set_vorticity(double *u, double *v, double *w, const Prm &prm) {
  for (int i = 1; i < prm.NX - 1; i++) {
    for (int j = 1; j < prm.NY - 1; j++) {
      W(i, j) = (V(i + 1, j) - V(i - 1, j)) / (2 * prm.dx) -
                (U(i, j + 1) - U(i, j - 1)) / (2 * prm.dy);
    }
  }
}

void buildPoissonMatrix(vector<Trip> &coeffs, const Prm &prm) {
  // Remember we are taking the following BC for the pressure:
  // on the top, bottom and left boundaries, the normal derivative of the
  // pressure is 0, i.e.:  P(i, NY - 1) = P(i, NY - 2)
  //                                                                                            P(i, 0) = P(i, 1);
  //                                                                                            P(0, j) = P(1, j);
  // respectively. On the right boundary, the pressure is 0, i.e.: P(NX - 1, j)
  // = -P(NX - 2, j). Also, note that the coefficients are ordered by column,
  // i.e. the first NY elements are the first column, the next NY elements are
  // the second column, and so on. The matrix has size (nx * ny) x (nx * ny),
  // where nx = NX - 2 and ny = NY - 2 (i.e. the number of points in the domain
  // minus the ghost points). Matrix for the 2nd derivative in x (except for the
  // division by dx^2)
  //  ------ ny ------
  // ||-----------------------------------------------------------------------------------------||
  // || -1           |  1           | || |
  // ||   -1         |    1         | || |
  // ||      .       |      .       | || ny
  // ||        .     |        .     | || |
  // ||          .   |          .   | || |
  // ||           -1 |            1 | || |
  // ||-----------------------------------------------------------------------------------------||
  // || 1            | -2           | 1            | ||
  // ||   1          |   -2         |   1          | ||
  // ||      .       |      .       |      .       | ||
  // ||        .     |        .     |        .     | ||
  // ||          .   |          .   |          .   | ||
  // ||            1 |           -2 |            1 | ||
  // ||-----------------------------------------------------------------------------------------||
  // ||              | 1            | -2           | 1            | ||
  // ||              |   1          |   -2         |   1          | ||
  // ||              |      .       |      .       |      .       | ||
  // ||              |        .     |        .     |        .     | ||
  // ||              |          .   |          .   |          .   | ||
  // ||              |            1 |           -2 |            1 | ||
  // ||-----------------------------------------------------------------------------------------||
  // ||                             |  .           |  .           |  . | ||
  // ||                             |    .         |    .         |    . | ||
  // ||                             |      .       |      .       |      . | ||
  // ||                             |        .     |        .     |        . |
  // ||
  // ||                             |          .   |          .   |          .
  // |              ||
  // ||                             |            . |            . |            .
  // |              ||
  // ||-----------------------------------------------------------------------------------------||
  // ||                                            | 1            | -2 | 1 ||
  // ||                                            |   1          |   -2 |   1
  // ||
  // ||                                            |      .       |      . | .
  // ||
  // ||                                            |        .     |        . |
  // .     ||
  // ||                                            |          .   |          .
  // |          .   ||
  // ||                                            |            1 |           -2
  // |            1 ||
  // ||-----------------------------------------------------------------------------------------||
  // ||                                                           | 1 | -3 ||
  // ||                                                           |   1 |   -3
  // ||
  // ||                                                           |      . | .
  // ||
  // ||                                                           |        . |
  // .     ||
  // ||                                                           |          .
  // |          .   ||
  // ||                                                           |            1
  // |           -3 ||
  // ||-----------------------------------------------------------------------------------------||

  // Matrix for the 2nd derivative in y (except for the division by dy^2)
  //  --------- ny --------
  // ||------------------------------------------------------------------------------------||
  // || -1  1              | || |
  // ||  1 -2  1           | || |
  // ||    1 -2  1         | || |
  // ||      .  .  .       | || ny
  // ||        .  .  .     | || |
  // ||          .  .  .   | || |
  // ||            1 -2  1 | || |
  // ||               1 -1 | || |
  // ||------------------------------------------------------------------------------------||
  // ||                    | -1  1              | ||
  // ||                    |  1 -2  1           | ||
  // ||                    |    1 -2  1         | ||
  // ||                    |      .  .  .       | ||
  // ||                    |        .  .  .     | ||
  // ||                    |          .  .  .   | ||
  // ||                    |            1 -2  1 | ||
  // ||                    |               1 -1 | ||
  // ||------------------------------------------------------------------------------------||
  // ||                                          |   .                | ||
  // ||                                          |     .              | ||
  // ||                                          |       .            | ||
  // ||                                          |         .          | ||
  // ||                                          |           .        | ||
  // ||                                          |             .      | ||
  // ||                                          |               .    | ||
  // ||                                          |                 .  | ||
  // ||------------------------------------------------------------------------------------||
  // ||                                                               | -1  1 ||
  // ||                                                               |  1 -2  1
  // ||
  // ||                                                               |    1 -2
  // 1         ||
  // ||                                                               |      .
  // .  .       ||
  // ||                                                               |        .
  // .  .     ||
  // ||                                                               | .  .  .
  // ||
  // ||                                                               | 1 -2  1
  // ||
  // ||                                                               | 1 -1 ||
  // ||------------------------------------------------------------------------------------||

  // number of non-zero coefficients in the Poi matrix
  // dx -> 3 * ny * (nx - 2) + 2 * (2 * ny) = ny * (3 * nx - 2)
  // dy -> (3 * ny - 2) * nx
  // common (diag) -> nx * ny
  // total = dx + dy - common = 5 * nx * ny - 2 * ny - 2 * nx

  // we will store the minus of the coefficients, because we will use the
  // Cholesky decomposition, which requires a positive definite matrix

  coeffs.reserve((uint)(5 * prm.nx * prm.ny - 2 * (prm.ny + prm.nx)));
  int dim = prm.nx * prm.ny;
  double dx_2 = 1. / (prm.dx * prm.dx);
  double dy_2 = 1. / (prm.dy * prm.dy);
  double diagX, diagY;

  for (int i = 0; i < dim; i++) {
    // diagonal
    diagX = -2. * dx_2;
    diagY = -2. * dy_2;
    if (i % prm.ny == 0 || i % prm.ny == prm.ny - 1)
      diagY = -dy_2;
    // for full neumann BC (test case, remember that in this case the matrix is
    // singular and the Cholesky decomposition will fail, you have to change for
    // example the component (0, 1) to 0, and, to make the matrix symmetric, the
    // component (1, 0) to 0) if (i < prm.ny || i >= dim - prm.ny) diagX =
    // -dx_2; for mixed dirichlet-neumann BC (our case)
    if (i < prm.ny)
      diagX = -dx_2;
    if (i >= dim - prm.ny)
      diagX = -3. * dx_2;
    coeffs.push_back(Trip(i, i, -diagX - diagY));

    // Dxx part (secondary diagonals)
    if (i < dim - prm.ny) {
      coeffs.push_back(Trip(i, i + prm.ny, -dx_2)); // upper diagonal
      coeffs.push_back(Trip(i + prm.ny, i, -dx_2)); // lower diagonal
    }

    // Dyy part (secondary diagonals)
    if ((i + 1) % prm.ny != 0) {
      coeffs.push_back(Trip(i, i + 1, -dy_2)); // upper diagonal
      coeffs.push_back(Trip(i + 1, i, -dy_2)); // lower diagonal
    }
  }
  // coeffs.push_back(Trip(0, 1, dy_2));
  // coeffs.push_back(Trip(1, 0, dy_2));
}
