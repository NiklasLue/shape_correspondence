import os
import sys
from shutil import copyfile
import numpy as np
import pandas as pd
from collections import defaultdict


def read_off(filepath):
    """
    read a standard .off file

    Parameters
    -------------------------
    file : path to a '.off'-format file

    Output
    -------------------------
    vertices,faces : (n,3), (m,3) array of vertices coordinates
                    and indices for triangular faces
    """
    with open(filepath, 'r') as f:
        if f.readline().strip() != 'OFF':
            raise TypeError('Not a valid OFF header')
        n_verts, n_faces, _ = [int(x) for x in f.readline().strip().split(' ')]
        vertices = [[float(x) for x in f.readline().strip().split()] for _ in range(n_verts)]
        if n_faces > 0:
            faces = [[int(x) for x in f.readline().strip().split()][1:4] for _ in range(n_faces)]
            faces = np.asarray(faces)
        else:
            faces = None

    return np.asarray(vertices), faces


def read_obj(filepath):
    """
    read a standard .obj file

    Parameters
    -------------------------
    file : path to a '.off'-format file

    Output
    -------------------------
    vertices,faces : (n,3), (m,3) array of vertices coordinates
                    and indices for triangular faces
    """
    with open(filepath, 'r') as f:

        vertices = []
        faces = []

        for line in f:
            line = line.strip()
            if line == '' or line[0] == '#':
                continue

            line = line.split()
            if line[0] == 'v':
                vertices.append([float(x) for x in line[1:]])
            elif line[0] == 'f':
                faces.append([int(x.split('/')[0]) - 1 for x in line[1:]])

    return np.asarray(vertices), np.asarray(faces)

def read_ply(filename, allow_bool=False):
    """ Read a .ply (binary or ascii) file and store the elements in pandas DataFrame.
    https://github.com/daavoo/pyntcloud/blob/master/pyntcloud/io/ply.py
    Parameters
    ----------
    filename: str
        Path to the filename
    allow_bool: bool
        flag to allow bool as a valid PLY dtype. False by default to mirror original PLY specification.
    Returns
    -------
    data: dict
        Elements as pandas DataFrames; comments and ob_info as list of string
    """
    sys_byteorder = ('>', '<')[sys.byteorder == 'little']

    ply_dtypes = dict([
        (b'int8', 'i1'),
        (b'char', 'i1'),
        (b'uint8', 'u1'),
        (b'uchar', 'b1'),
        (b'uchar', 'u1'),
        (b'int16', 'i2'),
        (b'short', 'i2'),
        (b'uint16', 'u2'),
        (b'ushort', 'u2'),
        (b'int32', 'i4'),
        (b'int', 'i4'),
        (b'uint32', 'u4'),
        (b'uint', 'u4'),
        (b'float32', 'f4'),
        (b'float', 'f4'),
        (b'float64', 'f8'),
        (b'double', 'f8')
    ])

    valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}
    if allow_bool:
        ply_dtypes[b'bool'] = '?'

    with open(filename, 'rb') as ply:

        if b'ply' not in ply.readline():
            raise ValueError('The file does not start with the word ply')
        # get binary_little/big or ascii
        fmt = ply.readline().split()[1].decode()
        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        line = []
        dtypes = defaultdict(list)
        count = 2
        points_size = None
        mesh_size = None
        has_texture = False
        comments = []
        while b'end_header' not in line and line != b'':
            line = ply.readline()

            if b'element' in line:
                line = line.split()
                name = line[1].decode()
                size = int(line[2])
                if name == "vertex":
                    points_size = size
                elif name == "face":
                    mesh_size = size

            elif b'property' in line:
                line = line.split()
                # element mesh
                if b'list' in line:

                    if b"vertex_indices" in line[-1] or b"vertex_index" in line[-1]:
                        mesh_names = ["n_points", "v1", "v2", "v3"]
                    else:
                        has_texture = True
                        mesh_names = ["n_coords"] + ["v1_u", "v1_v", "v2_u", "v2_v", "v3_u", "v3_v"]

                    if fmt == "ascii":
                        # the first number has different dtype than the list
                        dtypes[name].append(
                            (mesh_names[0], ply_dtypes[line[2]]))
                        # rest of the numbers have the same dtype
                        dt = ply_dtypes[line[3]]
                    else:
                        # the first number has different dtype than the list
                        dtypes[name].append(
                            (mesh_names[0], ext + ply_dtypes[line[2]]))
                        # rest of the numbers have the same dtype
                        dt = ext + ply_dtypes[line[3]]

                    for j in range(1, len(mesh_names)):
                        dtypes[name].append((mesh_names[j], dt))
                else:
                    if fmt == "ascii":
                        dtypes[name].append(
                            (line[2].decode(), ply_dtypes[line[1]]))
                    else:
                        dtypes[name].append(
                            (line[2].decode(), ext + ply_dtypes[line[1]]))

            elif b'comment' in line:
                line = line.split(b" ", 1)
                comment = line[1].decode().rstrip()
                comments.append(comment)

            count += 1

        # for bin
        end_header = ply.tell()

    data = {}

    if comments:
        data["comments"] = comments

    if fmt == 'ascii':
        top = count
        bottom = 0 if mesh_size is None else mesh_size

        names = [x[0] for x in dtypes["vertex"]]

        data["points"] = pd.read_csv(filename, sep=" ", header=None, engine="python",
                                     skiprows=top, skipfooter=bottom, usecols=names, names=names)

        for n, col in enumerate(data["points"].columns):
            data["points"][col] = data["points"][col].astype(
                dtypes["vertex"][n][1])

        if mesh_size :
            top = count + points_size

            names = np.array([x[0] for x in dtypes["face"]])
            usecols = [1, 2, 3, 5, 6, 7, 8, 9, 10] if has_texture else [1, 2, 3]
            names = names[usecols]

            data["mesh"] = pd.read_csv(
                filename, sep=" ", header=None, engine="python", skiprows=top, usecols=usecols, names=names)

            for n, col in enumerate(data["mesh"].columns):
                data["mesh"][col] = data["mesh"][col].astype(
                    dtypes["face"][n + 1][1])

    else:
        with open(filename, 'rb') as ply:
            ply.seek(end_header)
            points_np = np.fromfile(ply, dtype=dtypes["vertex"], count=points_size)
            if ext != sys_byteorder:
                points_np = points_np.byteswap().newbyteorder()
            data["points"] = pd.DataFrame(points_np)
            if mesh_size:
                mesh_np = np.fromfile(ply, dtype=dtypes["face"], count=mesh_size)
                if ext != sys_byteorder:
                    mesh_np = mesh_np.byteswap().newbyteorder()
                data["mesh"] = pd.DataFrame(mesh_np)
                data["mesh"].drop('n_points', axis=1, inplace=True)

    return data["points"], data["mesh"]

# def read_ply(ply_file):
#     with open(ply_file, 'r', encoding = "ISO-8859-1") as f:
#         lines = f.readlines()
#         verts_num = int(lines[2].split(' ')[-1])
#         faces_num = int(lines[6].split(' ')[-1])
#         verts_lines = lines[9:9 + verts_num]
#         faces_lines = lines[9 + verts_num:]
#         verts = np.array([list(map(float, l.strip().split(' '))) for l in verts_lines])
#         faces = np.array([list(map(int, l.strip().split(' '))) for l in faces_lines])[:,1:]

#     return verts, faces 


def write_off(filepath, vertices, faces, precision=None, face_colors=None):
    """
    Writes a .off file

    Parameters
    --------------------------
    filepath  : path to file to write
    vertices  : (n,3) array of vertices coordinates
    faces     : (m,3) array of indices of face vertices
    precision : int - number of significant digits to write for each float
    """
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0] if faces is not None else 0
    precision = precision if precision is not None else 16

    if face_colors is not None:
        assert face_colors.shape[0] == faces.shape[0], "PB"
        if face_colors.max() <= 1:
            face_colors = (256 * face_colors).astype(int)

    with open(filepath, 'w') as f:
        f.write('OFF\n')
        f.write(f'{n_vertices} {n_faces} 0\n')
        for i in range(n_vertices):
            f.write(f'{" ".join([f"{coord:.{precision}f}" for coord in vertices[i]])}\n')

        if n_faces != 0:
            for j in range(n_faces):
                if face_colors is None:
                    f.write(f'3 {" ".join([str(tri) for tri in faces[j]])}\n')
                else:
                    f.write(f'4 {" ".join([str(tri) for tri in faces[j]])} ')
                    f.write(f'{" ".join([str(tri_c) for tri_c in face_colors[j]])}\n')


def read_vert(filepath):
    """
    Read a .vert file from TOSCA dataset

    Parameters
    ----------------------
    filepath : path to file

    Output
    ----------------------
    vertices : (n,3) array of vertices coordinates
    """
    vertices = [[float(x) for x in line.strip().split()] for line in open(filepath, 'r')]
    return np.asarray(vertices)


def read_tri(filepath, from_matlab=True):
    """
    Read a .tri file from TOSCA dataset

    Parameters
    ----------------------
    filepath    : path to file
    from_matlab : whether file indexing starts at 1

    Output
    ----------------------
    faces : (m,3) array of vertices indices to define faces
    """
    faces = [[int(x) for x in line.strip().split()] for line in open(filepath,'r')]
    faces = np.asarray(faces)
    if from_matlab and np.min(faces) > 0:
        raise ValueError("Indexing starts at 0, can't set the from_matlab argument to True ")
    return faces - int(from_matlab)


def write_mtl(filepath, texture_im='texture_1.jpg'):
    """
    Writes a .mtl file for a .obj mesh

    Parameters
    ----------------------
    filepath   : path to file
    texture_im : name of the image of texture
    """
    with open(filepath, 'w') as f:
        f.write('newmtl material_0\n')
        f.write(f'Ka  {0.2:.6f} {0.2:.6f} {0.2:.6f}\n')
        f.write(f'Kd  {1.:.6f} {1.:.6f} {1.:.6f}\n')
        f.write(f'Ks  {1.:.6f} {1.:.6f} {1.:.6f}\n')
        f.write(f'Tr  {1:d}\n')
        f.write(f'Ns  {0:d}\n')
        f.write(f'illum {2:d}\n')
        f.write(f'map_Kd {texture_im}')


def _get_data_dir():
    """
    Return the directory where texture data is savec

    Output
    ---------------------
    data_dir : str - directory of texture data
    """
    curr_dir = os.path.dirname(__file__)
    return os.path.join(curr_dir,'data')


def get_uv(vertices, ind1, ind2, mult_const=1):
    """
    Extracts UV coordinates for a mesh for a .obj file

    Parameters
    -----------------------------
    vertices   : (n,3) coordinates of vertices
    ind1       : int - column index to use as first coordinate
    ind2       : int - column index to use as second coordinate
    mult_const : float - number of time to repeat the pattern

    Output
    ------------------------------
    uv : (n,2) UV coordinates of each vertex
    """
    vt = vertices[:,[ind1,ind2]]
    vt -= np.min(vt)
    vt = mult_const * vt / np.max(vt)
    return vt


def write_obj(filepath, vertices, faces, uv=None, mtl_file='material.mtl', texture_im='texture_1.jpg',
              precision=6, verbose=False):
    """
    Writes a .obj file with texture.
    Writes the necessary material and texture files.

    Parameters
    -------------------------
    filepath   : str - path to the .obj file to write
    vertices   : (n,3) coordinates of vertices
    faces      : (m,3) faces defined by vertex indices
    uv         : uv map for each vertex. If not specified no texture is used
    mtl_file   : str - name of the .mtl file
    texture_im : str - name of the .jpg file definig texture
    """
    use_texture = uv is not None
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0]
    precision = 16 if precision is None else precision

    dir_name = os.path.dirname(filepath)

    if use_texture:
        # Remove useless part of the path if written by mistake
        mtl_file = os.path.basename(mtl_file)
        texture_file = os.path.basename(texture_im)

        # Add extensions if forgotten
        if os.path.splitext(mtl_file)[1] != '.mtl':
            mtl_file += '.mtl'
        if os.path.splitext(texture_file)[1] != '.jpg':
            texture_file += '.jpg'

        # Write .mtl and .jpg files if necessary
        mtl_path = os.path.join(dir_name, mtl_file)
        texture_path = os.path.join(dir_name, texture_file)

        if not os.path.isfile(texture_path):
            data_texture = os.path.join(_get_data_dir(), texture_im)
            if not os.path.isfile(data_texture):
                raise ValueError(f"Texture {texture_im} does not exist")
            copyfile(data_texture, texture_path)
            print(f'Copy texture at {texture_path}')

        if not os.path.isfile(mtl_path):
            write_mtl(mtl_path,texture_im=texture_im)
            if verbose:
                print(f'Write material at {mtl_path}')

        # Write the .obj file
        mtl_name = os.path.splitext(mtl_file)[0]

    with open(filepath,'w') as f:
        if use_texture:
            f.write(f'mtllib ./{mtl_name}.mtl\ng\n')

        f.write(f'# {n_vertices} vertices - {n_faces} faces\n')
        for i in range(n_vertices):
            f.write(f'v {" ".join([f"{coord:.{precision}f}" for coord in vertices[i]])}\n')

        if use_texture and n_faces > 0:
            f.write(f'g {mtl_name}_export\n')
            f.write('usemtl material_0\n')

            for j in range(n_faces):
                f.write(f'f {" ".join([f"{1+tri:d}/{1+tri:d}" for tri in faces[j]])}\n')

            for k in range(n_vertices):
                f.write(f'vt {" ".join([str(coord) for coord in uv[k]])}\n')

        elif n_faces > 0:
            for j in range(n_faces):
                f.write(f'{" ".join(["f"] + [str(1+tri) for tri in faces[j]])}\n')

    if verbose:
        print(f'Write .obj file at {filepath}')
