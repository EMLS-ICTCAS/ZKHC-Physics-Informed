# 读取.npz文件
import numpy as np
import os

from fipy import  *
import fipy
from packaging.version import Version
fipy.meshes.gmshMesh._gmshVersion = lambda **kwargs : Version("2.0")

# --- 加载网格 ---
filename = "cylinder_mesh_0.5.msh2"  # 修改文件名以匹配
msh_filename = os.path.join("flow_dataset_2d_obstacle", filename)
mesh = Gmsh2D(msh_filename)

results_dir = os.path.join("flow_dataset_2d", "static_flow_datasets.npz")
data_file = np.load(results_dir, allow_pickle=True)
Vx = data_file["datasets"][0]["Vx"]

view = Viewer(Vx, title="X")
view.plot()