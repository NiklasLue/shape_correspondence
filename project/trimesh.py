import os

import trimesh

from pyFM.mesh import file_utils
from pyFM.mesh.trimesh import TriMesh
from .utils import read_ply
import robust_laplacian
import potpourri3d as pp3d
import scipy
import numpy as np

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
        
    def laplacian_spectrum(self, k, intrinsic=False, return_spectrum=True, robust=False, verbose=False):
        """
        Compute the Laplace Beltrami Operator and its spectrum.
        Consider using the .process() function for easier use !

        Parameters
        -------------------------
        K               : int - number of eigenvalues to compute
        intrinsic       : bool - Use intrinsic triangulation
        robust          : bool - use tufted laplacian
        return_spectrum : bool - Whether to return the computed spectrum

        Output
        -------------------------
        eigenvalues, eigenvectors : (k,), (n,k) - Only if return_spectrum is True.
        """
        eps = 1e-8
        
        if self.facelist is None:
            robust = True

        if robust:
            mollify_factor = 1e-5
        elif intrinsic:
            mollify_factor = 0

        if robust or intrinsic:
            self._intrinsic = intrinsic
            if self.facelist is not None:
                self.W, self.A = robust_laplacian.mesh_laplacian(self.vertlist, self.facelist, mollify_factor=mollify_factor)
            else:
                self.W, self.A = robust_laplacian.point_cloud_laplacian(self.vertlist, mollify_factor=mollify_factor)

        else:
            self.W = pp3d.cotan_laplacian(self.vertlist, self.facelist, denom_eps=1e-10)
            self.A = pp3d.vertex_areas(self.vertlist, self.facelist)
            self.A += eps * np.mean(self.A)
            self.A = scipy.sparse.diags(self.A)

        # If k is 0, stop here
        
        if k > 0:
            # Prepare matrices
            L_eigsh = (self.W + scipy.sparse.identity(self.W.shape[0]) * eps).tocsc()
            Mmat = self.A
            eigs_sigma = eps

            failcount = 0
            while True:
                try:
                    # We would be happy here to lower tol or maxiter since we don't need these to be super precise,
                    # but for some reason those parameters seem to have no effect
                    self.eigenvalues, self.eigenvectors = scipy.sparse.linalg.eigsh(
                        L_eigsh, k=k, M=Mmat, sigma=eigs_sigma
                    )

                    # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
                    self.eigenvalues = np.clip(self.eigenvalues, a_min=0.0, a_max=float("inf"))

                    break
                except Exception as e:
                    print(e)
                    if failcount > 3:
                        raise ValueError("failed to compute eigendecomp")
                    failcount += 1
                    print("--- decomp failed; adding eps ===> count: " + str(failcount))
                    
                
            if return_spectrum:
                return self.eigenvalues, self.eigenvectors