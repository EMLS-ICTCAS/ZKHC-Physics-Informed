import numpy as np
import h5py
import os
from defaults import cols_removed_beginning, cols_removed_end

# def read_data_file(file_path):
#     """Read data from file. The file is of the form:
#     header_0
#     x_00 x_01 ... x_0m
#     x_10 x_11 ... x_1m
#     ...
#     x_n0 x_n1 ... x_nm

#     header_1
#     x_00 x_01 ... x_0m
#     x_10 x_11 ... x_1m
#     ...
#     x_n0 x_n1 ... x_nm

#     ...

#     header_k
#     x_00 x_01 ... x_0m
#     x_10 x_11 ... x_1m
#     ...
#     x_n0 x_n1 ... x_nm
#     """
#     headers = []
#     data_blocks = []

#     with open(file_path, 'r') as file:
#         lines = file.readlines()

#     i = 0
#     while i < len(lines):
#         line = lines[i].strip()
#         if line:
#             header = float(line)
#             headers.append(header)

#             i += 1
#             block_lines = []
#             while i < len(lines) and lines[i].strip():
#                 aux = list(map(float, lines[i].split()))
#                 block_lines.append(aux)
#                 i += 1
#             data_block = np.array(block_lines)
#             data_blocks.append(data_block)
#         i += 1

#     headers_array = np.array(headers)
#     data_blocks_array = np.array(data_blocks)

#     return headers_array, data_blocks_array


def get_num_frames(folder_path: str) -> int:
    num_frames = len(
        [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ]
    )
    return num_frames


def read_data_object(file_path: str):
    """
    Read data from file. The file is of the form:

    A1 x0
    A2 x1
    ...
    An xn
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        header, value = line.split()
        data.append(float(value))

    data_array = np.array(data)

    return data_array


def readSetupFromHDF5(hdf5_file: str):
    with h5py.File(hdf5_file, "r") as file:
        # Accessing the first element and converting to scalar
        Re = file["Re"][()].item()
        NX = int(file["NX"][()].item())
        NY = int(file["NY"][()].item())
        LX = file["LX"][()].item()
        LY = file["LY"][()].item()
        L = file["L"][()].item()
        U = file["U"][()].item()
        nu = file["nu"][()].item()
        dx = file["dx"][()].item()
        dy = file["dy"][()].item()
        dt = file["dt"][()].item()
        T = file["T"][()].item()
        obstacle = file["obstacle"][()].item().decode("utf-8")
        w_on = bool(file["w_on"][()].item())
        animation_on = bool(file["animation_on"][()].item())
    return Re, NX, NY, LX, LY, L, U, nu, dx, dy, dt, T, obstacle, w_on, animation_on


def readSolutionFromHDF5(hdf5_file: str):
    with h5py.File(hdf5_file, "r") as file:
        # NX = int(file['NX'][()].item())
        # NY = int(file['NY'][()].item())
        u = file["u"][:]
        v = file["v"][:]
        w = file["w"][:]
        p = file["p"][:]
        t = file["t"][()].item()
    return t, u, v, w, p


def set_data(folder_path: str):
    num_frames = get_num_frames(folder_path)

    times = []
    data_blocks_u = []
    data_blocks_v = []
    data_blocks_w = []
    data_blocks_p = []
    for i in range(num_frames):
        file_path = folder_path + "sol_" + str(i) + ".h5"
        t, u, v, w, p = readSolutionFromHDF5(file_path)
        times.append(t)
        # remove ghost cells
        data_blocks_u.append(u[1:-1, 1:-1])
        data_blocks_v.append(v[1:-1, 1:-1])
        data_blocks_w.append(w[1:-1, 1:-1])
        data_blocks_p.append(p[1:-1, 1:-1])

    times = np.array(times)
    data_blocks_u = np.array(data_blocks_u)
    data_blocks_v = np.array(data_blocks_v)
    data_blocks_w = np.array(data_blocks_w)
    data_blocks_p = np.array(data_blocks_p)

    arrays_to_process = [data_blocks_u, data_blocks_v, data_blocks_w, data_blocks_p]

    # Iterate over each array and remove columns
    for i, data_array in enumerate(arrays_to_process):
        arrays_to_process[i] = np.delete(
            data_array, np.s_[:cols_removed_beginning], axis=1
        )
        arrays_to_process[i] = np.delete(
            arrays_to_process[i], np.s_[-cols_removed_end:], axis=1
        )

        tmp = arrays_to_process[i].copy()

        arrays_to_process[i] = np.zeros((len(tmp), tmp.shape[2], tmp.shape[1]))

        # transpose the data and reverse
        for j, data in enumerate(tmp):
            arrays_to_process[i][j] = data.T[::-1]

    # Update the original arrays with the modified ones
    data_blocks_u, data_blocks_v, data_blocks_w, data_blocks_p = arrays_to_process

    return times, data_blocks_u, data_blocks_v, data_blocks_w, data_blocks_p
