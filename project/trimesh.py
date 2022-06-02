import os

import trimesh

from pyFM.mesh import file_utils
from pyFM.mesh.trimesh import TriMesh
from .utils import read_ply

class TriMeshPly(TriMesh):
    """
    Mesh Class (can also represent point clouds)
    ________

    Attributes
    ------------------
    # FILE INFO
    path         : str - path the the loaded .off file. Set to None if the geometry is modified.
    meshname     : str - name of the .off file. Remains even when geometry is modified. '_n' is
                   added at the end if the mesh was normalized.

    # GEOMETRY
    vertlist       : (n,3) array of n vertices coordinates
    facelist       : (m,3) array of m triangle indices
    normals        : (m,3) array of normals
    vertex_normals : (n,3) array of vertex normals
                     (change weighting type with self.set_vertex_normal_weighting)

    # SPECTRAL INFORMATION
    W            : (n,n) sparse cotangent weight matrix
    A            : (n,n) sparse area matrix (either diagonal or computed with finite elements)
    eigenvalues  : (K,) eigenvalues of the Laplace Beltrami Operator
    eigenvectors : (n,K) eigenvectors of the Laplace Beltrami Operator

    Properties
    ------------------
    area         : float - area of the mesh
    face_areas   : (m,) per face area
    vertex_areas : (n,) per vertex area
    center_mass  : (3,) center of mass

    n_vertices   : int - number of vertices
    n_faces      : int - number of faces
    edges        : (p,2) edges defined by vertex indices
    """
    def __init__(self, *args, **kwargs):
        super(TriMeshPly, self).__init__(*args, **kwargs)

    def barycentric_to_points(self, b, triangle_id):
        """
        calculates euclidean coordinates from given barycentric coordinates and triangles
        :param b: barycentric coordinates (n, 3)
        :param triangle_id: list of triangle ID's corresponding to the barycentric coordinates (n)

        Returns
        ---------
        p : (n, 3) float
          Point coordinates in euclidean space
        """

        if self._triangle_vertices is None:
            self._init_triangles()

        p = trimesh.triangles.barycentric_to_points(self._triangle_vertices[triangle_id], b)

        return p

    def _load_mesh(self, meshpath):
        """
        Load a mesh from a file

        Parameters:
        --------------------------
        meshpath : path to file
        """

        if os.path.splitext(meshpath)[1] == '.off':
            self.vertlist, self.facelist = file_utils.read_off(meshpath)
        elif os.path.splitext(meshpath)[1] == '.obj':
            self.vertlist, self.facelist = file_utils.read_obj(meshpath)
        elif os.path.splitext(meshpath)[1] == '.ply':
            self.vertlist, self.facelist = read_ply(meshpath)

        else:
            raise ValueError('Provide file in .off, .obj or .ply format')

        self.path = meshpath
        self.meshname = os.path.splitext(os.path.basename(meshpath))[0]

    def _init_triangles(self):

        print("Initializing list of triangle vertices...")
        self._triangle_vertices = self.vertlist[self.facelist]


    def _init_all_attributes(self):
        super()._init_all_attributes()

        self._triangle_vertices = None