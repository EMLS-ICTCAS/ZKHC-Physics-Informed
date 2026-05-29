import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import h5py
import time
import os
from dataclasses import dataclass, field
from typing import List, Tuple
from tqdm import tqdm

# ================= 1. 参数配置 (对应 misc.hpp) =================
@dataclass
class Prm:
    """模拟参数结构体"""
    LX: float = 2.0
    LY: float = 1.0
    nx: int = 200
    ny: int = 100
    dt: float = 0.5     # 0.005
    T: float = 5.0      #50.0
    Re: float = 100.0
    nu: float = 0.001
    L: float = 0.1
    obstacle_ON: bool = True
    
    # 派生参数
    NX: int = field(init=False)
    NY: int = field(init=False)
    dx: float = field(init=False)
    dy: float = field(init=False)
    NXNY: int = field(init=False)
    nxny: int = field(init=False)
    U: float = field(init=False)

    def __post_init__(self):
        self.NX = self.nx + 2
        self.NY = self.ny + 2
        self.dx = self.LX / self.nx
        self.dy = self.LY / self.ny
        self.NXNY = self.NX * self.NY
        self.nxny = self.nx * self.ny
        self.U = self.Re * self.nu / self.L

def x_coord(i: int, prm: Prm) -> float:
    return -0.5 * prm.dx + i * prm.dx

def y_coord(j: int, prm: Prm) -> float:
    return -0.5 * prm.dy + j * prm.dy

# ================= 2. 浸没边界障碍物 (对应 object.hpp) =================
TOL = 1e-8

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Grid:
    i: int
    j: int

class Object:
    def __init__(self, prm: Prm):
        self.prm = prm
        NX, NY = prm.NX, prm.NY
        self.IsInside = np.zeros((NX, NY), dtype=bool)
        self.IsGhost = np.zeros((NX, NY), dtype=bool)
        self.IsInterface = np.zeros((NX, NY), dtype=bool)
        self.BoundaryPoints = np.zeros((NX, NY, 2), dtype=float)  # [x, y]
        self.GhostPoints: List[Grid] = []
        self.MirrorPoints: List[Point] = []
        self.InterpolatingPoints: List[List[Grid]] = []
        self.count_ghost = 0
        self._init()

    def _init(self):
        prm = self.prm
        # 1. 标记内部点与边界最近点
        for i in range(prm.NX):
            for j in range(prm.NY):
                self.IsInside[i, j] = self.is_inside(i, j, prm)
                self.BoundaryPoints[i, j] = self.closest_boundary_point(x_coord(i, prm), y_coord(j, prm))

        # 2. 标记幽灵点
        count_gh = 0
        for i in range(prm.NX):
            for j in range(prm.NY):
                if self.IsInside[i, j]:
                    has_out = (not self.IsInside[max(0,i-1), j] or not self.IsInside[min(prm.NX-1,i+1), j] or
                               not self.IsInside[i, max(0,j-1)] or not self.IsInside[i, min(prm.NY-1,j+1)])
                    self.IsGhost[i, j] = has_out
                    if has_out: count_gh += 1

        self.count_ghost = count_gh
        self.GhostPoints = [Grid(0,0)] * count_gh
        self.MirrorPoints = [Point(0,0)] * count_gh
        self.InterpolatingPoints = [[] for _ in range(count_gh)]

        # 3. 计算镜像点与插值模板
        count = 0
        for i in range(prm.NX):
            for j in range(prm.NY):
                if count >= count_gh: break
                if self.IsGhost[i, j]:
                    self.GhostPoints[count] = Grid(i, j)
                    self.MirrorPoints[count] = self.mirror_point(x_coord(i, prm), y_coord(j, prm))
                    self._set_interpolating_points(i, j, self.MirrorPoints[count], count, prm)
                    count += 1

        # 4. 标记界面点
        for i in range(prm.NX):
            for j in range(prm.NY):
                if not self.IsInside[i, j]:
                    has_in = (self.IsInside[max(0,i-1), j] or self.IsInside[min(prm.NX-1,i+1), j] or
                              self.IsInside[i, max(0,j-1)] or self.IsInside[i, min(prm.NY-1,j+1)])
                    self.IsInterface[i, j] = has_in

    def is_inside(self, i: int, j: int, prm: Prm) -> bool: raise NotImplementedError
    def closest_boundary_point(self, x: float, y: float) -> Point: raise NotImplementedError

    def mirror_point(self, x: float, y: float) -> Point:
        p_x, p_y = self.closest_boundary_point(x, y)
        return Point(2 * p_x - x, 2 * p_y - y)

    def sign_nx(self, x: float, mx: float) -> int:
        return 1 if mx - x > TOL else (-1 if mx - x < -TOL else 0)
    def sign_ny(self, y: float, my: float) -> int:
        return 1 if my - y > TOL else (-1 if my - y < -TOL else 0)

    def _set_interpolating_points(self, i: int, j: int, mirror: Point, count: int, prm: Prm):
        I, J = i - 2, j - 2
        while x_coord(I + 1, prm) < mirror.x + TOL: I += 1
        while y_coord(J + 1, prm) < mirror.y + TOL: J += 1

        sgn_nx = self.sign_nx(x_coord(i, prm), mirror.x)
        sgn_ny = self.sign_ny(y_coord(j, prm), mirror.y)

        if sgn_nx > 0: x1,x2,x5,x6_1,x6_2 = I+1, I+2, I+1, I+1, I+2
        else:
            I += 1
            x1,x2,x5,x6_1,x6_2 = I-1, I-2, I-1, I-1, I-2
        x3, x4 = I, I

        if sgn_ny > 0: y3,y4,y5,y6_1,y6_2 = J+1, J+2, J+1, J+2, J+1
        else:
            J += 1
            y3,y4,y5,y6_1,y6_2 = J-1, J-2, J-1, J-2, J-1
        y1, y2 = J, J

        d1 = (x_coord(x6_1, prm)-mirror.x)**2 + (y_coord(y6_1, prm)-mirror.y)**2
        d2 = (x_coord(x6_2, prm)-mirror.x)**2 + (y_coord(y6_2, prm)-mirror.y)**2
        x6, y6 = (x6_1, y6_1) if d1 < d2 else (x6_2, y6_2)

        clip = lambda idx, lim: max(0, min(lim-1, idx))
        self.InterpolatingPoints[count] = [
            Grid(clip(x1, prm.NX), clip(y1, prm.NY)), Grid(clip(x2, prm.NX), clip(y2, prm.NY)),
            Grid(clip(x3, prm.NX), clip(y3, prm.NY)), Grid(clip(x4, prm.NX), clip(y4, prm.NY)),
            Grid(clip(x5, prm.NX), clip(y5, prm.NY)), Grid(clip(x6, prm.NX), clip(y6, prm.NY))
        ]

class Circle(Object):
    def __init__(self, x0: float, y0: float, R: float, prm: Prm):
        self.x0, self.y0, self.R = x0, y0, R
        super().__init__(prm)
    def is_inside(self, i: int, j: int, prm: Prm) -> bool:
        return (x_coord(i, prm)-self.x0)**2 + (y_coord(j, prm)-self.y0)**2 < self.R**2
    def closest_boundary_point(self, x: float, y: float) -> Point:
        d = np.hypot(x-self.x0, y-self.y0)
        return [self.x0 + self.R*(x-self.x0)/d, self.y0 + self.R*(y-self.y0)/d]

# ================= 3. NS 求解器核心 (对应 NS.cpp) =================
def semilag(u: np.ndarray, v: np.ndarray, q: np.ndarray, prm: Prm, sign: int, obstacle: Object) -> np.ndarray:
    aux = q.copy()
    i_sl = slice(1, prm.NX-1)
    j_sl = slice(1, prm.NY-1)
    
    ui = u[i_sl, j_sl]; vi = v[i_sl, j_sl]; qi = q[i_sl, j_sl]
    mask = obstacle.IsInside[i_sl, j_sl] if prm.obstacle_ON else np.zeros_like(qi, dtype=bool)
    
    cond_u = (sign * ui > 0)
    a = np.where(cond_u, 1 - sign*ui*prm.dt/prm.dx, 1 + sign*ui*prm.dt/prm.dx)
    sign_u = np.where(cond_u, 1, -1)
    
    cond_v = (sign * vi > 0)
    b = np.where(cond_v, 1 - sign*vi*prm.dt/prm.dy, 1 + sign*vi*prm.dt/prm.dy)
    sign_v = np.where(cond_v, 1, -1)
    
    I, J = np.meshgrid(np.arange(1, prm.NX-1), np.arange(1, prm.NY-1), indexing='ij')
    q00 = q[I, J]; q10 = q[I - sign_u, J]; q01 = q[I, J - sign_v]; q11 = q[I - sign_u, J - sign_v]
    
    res = a*b*q00 + (1-a)*b*q10 + a*(1-b)*q01 + (1-a)*(1-b)*q11
    aux[i_sl, j_sl] = res

    for i in range(prm.NX):
        for j in range(prm.NY):
            if prm.obstacle_ON and obstacle.is_inside(i, j, prm):
                aux[i, j] = q[i, j]

    return aux

def semilag2(u: np.ndarray, v: np.ndarray, q0: np.ndarray, prm: Prm, obstacle: Object) -> np.ndarray:
    q1 = semilag(u, v, q0.copy(), prm, 1, obstacle)
    q1 = semilag(u, v, q1, prm, -1, obstacle)
    q1 = q0 + (q0 - q1) / 2.0
    return semilag(u, v, q1, prm, 1, obstacle)

def bc_velocity(u: np.ndarray, v: np.ndarray, prm: Prm) -> None:
    u[:, 0] = u[:, 1]; v[:, 0] = -v[:, 1]
    u[:, -1] = u[:, -2]; v[:, -1] = -v[:, -2]
    u[0, :] = 2.0 - u[1, :]; v[0, :] = -v[1, :]
    u[-1, :] = u[-2, :]; v[-1, :] = v[-2, :]

def bc_pressure(p: np.ndarray, prm: Prm) -> None:
    p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]
    p[0, :] = p[1, :]; p[-1, :] = -p[-2, :]

def set_vorticity(u: np.ndarray, v: np.ndarray, w: np.ndarray, prm: Prm) -> None:
    w[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*prm.dx) - \
                    (u[1:-1, 2:] - u[1:-1, :-2]) / (2*prm.dy)

def build_poisson_matrix(prm: Prm) -> sparse.csc_matrix:
    dim = prm.nx * prm.ny
    dx2, dy2 = 1.0/prm.dx**2, 1.0/prm.dy**2
    rows, cols, vals = [], [], []
    
    for i in range(dim):
        diagX, diagY = -2.0*dx2, -2.0*dy2
        if i % prm.ny == 0 or i % prm.ny == prm.ny - 1: diagY = -dy2
        if i < prm.ny: diagX = -dx2
        if i >= dim - prm.ny: diagX = -3.0*dx2
        
        rows.append(i); cols.append(i); vals.append(-diagX - diagY)
        if i < dim - prm.ny:
            rows.extend([i, i+prm.ny]); cols.extend([i+prm.ny, i]); vals.extend([-dx2, -dx2])
        if (i + 1) % prm.ny != 0:
            rows.extend([i, i+1]); cols.extend([i+1, i]); vals.extend([-dy2, -dy2])
            
    return sparse.coo_matrix((vals, (rows, cols)), shape=(dim, dim)).tocsc()

# ================= 4. write.hpp =================
def format_time_us(time_us: int) -> str:
    """对应 write.hpp::print，自动转换 μs/ms/s/min/h"""
    if time_us > 10000000:
        hours = time_us / 3.6e9
        if hours > 100: return f"{hours:.2f} h"
        mins = time_us / 6e7
        return f"{mins:.2f} min" if mins > 100 else f"{time_us/1e6:.2f} s"
    elif time_us > 10000:
        return f"{time_us/1e3:.2f} ms"
    return f"{time_us:.2f} μs"

def save_setup_hdf5(prm: Prm, object_type: str, vorticity_on: bool, animation_on: bool, filepath: str = "output/setup.h5"):
    """对应 write.hpp::saveSetupToHDF5"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, 'w') as f:
        for name, val in {"Re": prm.Re, "NX": prm.NX, "NY": prm.NY, "LX": prm.LX, "LY": prm.LY,
                          "L": prm.L, "U": prm.U, "nu": prm.nu, "dx": prm.dx, "dy": prm.dy,
                          "dt": prm.dt, "T": prm.T, "w_on": float(vorticity_on), "animation_on": float(animation_on)}.items():
            f.create_dataset(name, data=float(val))
        f.create_dataset("obstacle", data=object_type.encode('utf-8'))

def save_data_hdf5(plot_count: int, u: np.ndarray, v: np.ndarray, w: np.ndarray, p: np.ndarray, 
                   Nx: int, Ny: int, t: float, base_dir: str = "output/results"):
    """对应 write.hpp::saveDataToHDF5，每帧生成独立文件 sol_X.h5"""
    os.makedirs(base_dir, exist_ok=True)
    filepath = os.path.join(base_dir, f"sol_{plot_count}.h5")
    with h5py.File(filepath, 'w') as f:
        f.create_dataset("u", data=u, dtype='f8')
        f.create_dataset("v", data=v, dtype='f8')
        f.create_dataset("w", data=w, dtype='f8')
        f.create_dataset("p", data=p, dtype='f8')
        f.create_dataset("t", data=float(t))

# ================= 4. 主程序与 HDF5 I/O (对应 main.cpp) =================
def run_simulation(prm: Prm, object_type: str = "circle"):
    print(f"⚙ 初始化参数: Re={prm.Re}, nx={prm.nx}, ny={prm.ny}, T={prm.T}")
    
    # 创建障碍物
    if object_type == "circle":
        obstacle = Circle(x0=0.4, y0=0.5, R=0.05, prm=prm)
    else:
        raise ValueError("仅实现圆柱障碍物，可扩展其他类型")
        
    # 分配场
    u = np.ones((prm.NX, prm.NY))
    v = np.zeros((prm.NX, prm.NY))
    p = np.zeros((prm.NX, prm.NY))
    w = np.zeros((prm.NX, prm.NY))
    adv_u = np.zeros_like(u); adv_v = np.zeros_like(v)
    ustar = np.zeros_like(u); vstar = np.zeros_like(v)

    # HDF5 设置
    # h5f = h5py.File("karman_dataset.h5", 'w')
    # h5f.attrs.update(vars(prm))
    # num_frames = int(prm.T / 0.5) + 2
    # chunk = (1, prm.NX, prm.NY)
    # ds_u = h5f.create_dataset("u", shape=(num_frames, *chunk[1:]), chunks=chunk, dtype='f4')
    # ds_v = h5f.create_dataset("v", shape=(num_frames, *chunk[1:]), chunks=chunk, dtype='f4')
    # ds_w = h5f.create_dataset("w", shape=(num_frames, *chunk[1:]), chunks=chunk, dtype='f4')
    # ds_p = h5f.create_dataset("p", shape=(num_frames, *chunk[1:]), chunks=chunk, dtype='f4')
    # ds_t = h5f.create_dataset("time", shape=(num_frames,), dtype='f4')
    
    # 构建泊松矩阵并预求解器
    A = build_poisson_matrix(prm)
    plot_count = 0
    plot_dt = 0.5
    t = 0.0
    EPS = 1e-8
    fraction_completed = prm.T / 100.0
    # def write_frame():
    #     ds_u[plot_count] = u; ds_v[plot_count] = v
    #     ds_w[plot_count] = w; ds_p[plot_count] = p
    #     ds_t[plot_count] = t
        
    # write_frame()
    save_setup_hdf5(prm, object_type, vorticity_on=True, animation_on=True)
    save_data_hdf5(0, u, v, w, p, prm.NX, prm.NY, 0.0)
    plot_count = 1
    
    print("⏱ 开始时间步进...")
    pbar = tqdm(total=prm.T, desc="Simulating", unit="s")
    t_start = time.time()
    
    while t < prm.T - EPS:
        if prm.dt < EPS:
            print("⚠ 时间步过小，终止模拟"); break
            
        # 自适应 CFL
        max_uv = max(np.max(np.abs(u)), np.max(np.abs(v)))
        if max_uv > 1e-3:
            prm.dt = min(prm.dt, prm.dx * prm.dy / (2 * max_uv * (prm.dx + prm.dy)))
            
        # 平流
        adv_u[:] = semilag2(u, v, u, prm, obstacle)
        adv_v[:] = semilag2(u, v, v, prm, obstacle)
        
        # 扩散 + 预测速度
        lap_u = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / prm.dx**2 + \
                (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / prm.dy**2
        lap_v = (np.roll(v, -1, axis=0) - 2*v + np.roll(v, 1, axis=0)) / prm.dx**2 + \
                (np.roll(v, -1, axis=1) - 2*v + np.roll(v, 1, axis=1)) / prm.dy**2
                
        ustar = u + prm.dt * (adv_u + lap_u / prm.Re)
        vstar = v + prm.dt * (adv_v + lap_v / prm.Re)
        if prm.obstacle_ON:
            ustar[obstacle.IsInside] = 0.0
            vstar[obstacle.IsInside] = 0.0
            
        bc_velocity(ustar, vstar, prm)
        
        # 散度 (内部节点)
        div = np.zeros((prm.nx, prm.ny))
        div[:] = -(ustar[2:, 1:-1] - ustar[:-2, 1:-1]) / (2*prm.dx) - \
                 (vstar[1:-1, 2:] - vstar[1:-1, :-2]) / (2*prm.dy)
        if prm.obstacle_ON:
            div[obstacle.IsInside[1:-1, 1:-1]] = 0.0
            
        # 求解压力泊松方程
        p_solved = spsolve(A, div.ravel())
        p[1:-1, 1:-1] = p_solved.reshape((prm.nx, prm.ny))
        bc_pressure(p, prm)
        
        # 投影步
        u[1:-1, 1:-1] = ustar[1:-1, 1:-1] - (p[2:, 1:-1] - p[:-2, 1:-1]) / (2*prm.dx)
        v[1:-1, 1:-1] = vstar[1:-1, 1:-1] - (p[1:-1, 2:] - p[1:-1, :-2]) / (2*prm.dy)
        bc_velocity(u, v, prm)
        
        t += prm.dt
        
        # 输出检查
        if t >= (plot_count) * plot_dt - EPS:
            set_vorticity(u, v, w, prm)
            save_data_hdf5(plot_count, u, v, w, p, prm.NX, prm.NY, t)
            plot_count += 1
            print(f"📊 t={t:.2f}, dt={prm.dt:.5f}, plot={plot_count}")

        pbar.update(prm.dt)
    
    pbar.close()
    # h5f.close()
    print(f"✅ 模拟完成。总耗时: {time.time()-t_start:.2f}s | 数据已保存至 karman_dataset.h5")

if __name__ == "__main__":
    prm = Prm(LX=8.0, LY=1.0, nx=800, ny=100, Re=500, nu=1.506e-5, L=0.1, T=20.0, dt=0.1, obstacle_ON=True)
    run_simulation(prm, object_type="circle")
