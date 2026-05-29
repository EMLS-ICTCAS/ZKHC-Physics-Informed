import gmsh
import os

# 网格参数
mesh_size_obs = 0.02              # 障碍物附近网格尺寸
mesh_size_far = 0.2               # 远场网格尺寸
Lx, Ly = 10.0, 5.0               # 腔体尺寸
# 障碍物参数：圆心位于域中心
ObCx, ObCy, ObR = 5.0, 2.5, 0.5  # 圆心坐标与半径

# 输出目录
output_dir = "flow_dataset_2d_obstacle"
# output_dir = "flow_dataset_2d"
os.makedirs(output_dir, exist_ok=True)
# 输出文件
msh_file_name = "cylinder_mesh_0.5.msh2"
def generate_mesh():
    """
    使用 Gmsh 生成包含圆形障碍物的封闭腔体网格
    返回 .msh 文件路径
    """

    gmsh.initialize()
    gmsh.model.add("cavity_cylinder")

    # 全局网格尺寸
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_far)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_obs)

    # 矩形腔体
    rect = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    # 圆形障碍物
    disk = gmsh.model.occ.addDisk(ObCx, ObCy, 0, ObR, ObR)
    # 流体域 = 矩形 - 圆形（布尔差）
    fluid, _ = gmsh.model.occ.cut([(2, rect)], [(2, disk)])
    gmsh.model.occ.synchronize()

    # 获取流体域实体
    fluid_entities = [e for e in fluid if e[0] == 2]
    if not fluid_entities:
        raise RuntimeError("布尔运算失败，未生成二维流体域。")
    fluid_tag = fluid_entities[0][1]
    
    # 获取所有边界（1D 边）
    boundaries = gmsh.model.getBoundary([(2, fluid_tag)], combined=False)

    # 分类边界：根据质心坐标
    inlet = []
    outlet = []
    top = []
    bottom = []
    obstacle = []

    for dim, tag in boundaries:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        xc, yc, _ = com
        if abs(xc) < 1e-6:
            inlet.append(tag)
        elif abs(xc - Lx) < 1e-6:
            outlet.append(tag)
        elif abs(yc) < 1e-6:
            bottom.append(tag)
        elif abs(yc - Ly) < 1e-6:
            top.append(tag)
        else:
            obstacle.append(tag)

    # 创建物理组
    def add_phys_group(name, dim, tags):
        if tags:
            pg = gmsh.model.addPhysicalGroup(dim, tags)
            gmsh.model.setPhysicalName(dim, pg, name)

    add_phys_group("inlet", 1, inlet)
    add_phys_group("outlet", 1, outlet)
    add_phys_group("top", 1, top)
    add_phys_group("bottom", 1, bottom)
    add_phys_group("obstacle", 1, obstacle)
    gmsh.model.addPhysicalGroup(2, [fluid_tag], name="fluid")

    # 网格生成
    # gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    # gmsh.option.setNumber("Mesh.SaveAll", 1)
    # gmsh.option.setNumber("Mesh.Format", 1)
    gmsh.model.mesh.generate(2)

    msh_file = os.path.join(output_dir, msh_file_name)
    gmsh.write(msh_file)
    gmsh.finalize()
    return msh_file


if __name__ == '__main__':

    msh_file = generate_mesh()