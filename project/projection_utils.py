import time

import numpy as np
from tqdm.auto import tqdm

from pyFM.spectral.projection_utils import compute_lmax, compute_Deltamin, compute_all_dmin, project_to_mesh, barycentric_to_precise


def project_pc_to_triangles(vert_emb, faces, points_emb, precompute_dmin=True, batch_size=None, n_jobs=1, return_vec=False, verbose=False):
    """
    Project a pointcloud on a set of triangles in p-dimension. Projection is defined as
    barycentric coordinates on one of the triangle.
    Line i for the output has 3 non-zero values at indices j,k and l of the vertices of the
    triangle point i zas projected on.

    Parameters
    ----------------------------
    vert_emb        : (n1, p) coordinates of the mesh vertices
    faces           : (m1, 3) faces of the mesh defined as indices of vertices
    points_emb      : (n2, p) coordinates of the pointcloud
    precompute_dmin : Whether to precompute all the values of delta_min.
                      Faster but heavier in memory.
    batch_size      : If precompute_dmin is False, projects batches of points on the surface
    n_jobs          : number of parallel process for nearest neighbor precomputation
    return_vec      : If true, return a (n2, 3) vector with barycentric coordinates instead
                      of a sparse matrix.


    Output
    ----------------------------
    precise_map : (n2,n1) - precise point to point map.
    """
    if batch_size is not None:
        batch_size = None if batch_size < 2 else batch_size
    n_points = points_emb.shape[0]
    n_vertices = vert_emb.shape[0]

    face_match = np.zeros(n_points, dtype=int)
    bary_coord = np.zeros((n_points, 3))

    if verbose:
        print('Precompute edge lengths...')
        start_time = time.time()
    lmax = compute_lmax(vert_emb, faces)  # (n1,)
    if verbose:
        print(f'\tDone in {time.time()-start_time:.2f}s')

    if verbose:
        print('Precompute nearest vertex...')
        start_time = time.time()
    Deltamin = compute_Deltamin(vert_emb, points_emb, n_jobs=n_jobs)  # (n2,)
    if verbose:
        print(f'\tDone in {time.time()-start_time:.2f}s')

    dmin = None
    if precompute_dmin:
        if verbose:
            print('Precompute nearest vertex in each face...')
            start_time = time.time()
        dmin = compute_all_dmin(vert_emb, faces, points_emb)  # (n_f1,n2)
        dmin_params = None
        if verbose:
            print(f'\tDone in {time.time()-start_time:.2f}s')

    else:
        vert_sqnorms = np.linalg.norm(vert_emb, axis=1)**2
        points_sqnorm = np.linalg.norm(points_emb, axis=1)**2
        dmin_params = {
                       'vert_sqnorms': vert_sqnorms,
                       'points_sqnorm': points_sqnorm
                       }

    # Iterate along all points
    if precompute_dmin or batch_size is None:
        iterable = range(n_points) if not verbose else tqdm(range(n_points))
        # for vertind in tqdm(range(n2)):
        for vertind in iterable:
            faceind, bary = project_to_mesh(vert_emb, faces, points_emb, vertind, lmax, Deltamin,
                                            dmin=dmin, dmin_params=dmin_params)
            face_match[vertind] = faceind
            bary_coord[vertind] = bary

    else:
        n_batches = n_points // batch_size + int((n_points % batch_size) > 0)
        iterable = range(n_batches) if not verbose else tqdm(range(n_batches))

        for batchind in iterable:
            batch_minmax = [batch_size*batchind, min(n_points, batch_size*(1+batchind))]
            # print(batch_minmax)
            dmin_batch = compute_all_dmin(vert_emb, faces, points_emb[batch_minmax[0]:batch_minmax[1]],
                                          vert_sqnorm=vert_sqnorms, points_sqnorm=points_sqnorm[batch_minmax[0]:batch_minmax[1]])

            batch_iterable = range(*batch_minmax)  #if not verbose else tqdm(range(*batch_minmax))
            for vertind in batch_iterable:
                batch_vertind = vertind - batch_minmax[0]
                faceind, bary = project_to_mesh(vert_emb, faces, points_emb[batch_minmax[0]:batch_minmax[1]],
                                                batch_vertind, lmax, Deltamin[batch_minmax[0]:batch_minmax[1]],
                                                dmin=dmin_batch, dmin_params=dmin_params)

                face_match[vertind] = faceind
                bary_coord[vertind] = bary

    if return_vec:
        return bary_coord, face_match

    return barycentric_to_precise(faces, face_match, bary_coord, n_vertices=n_vertices)


def project_point_to_rep_to_rep(mesh1, mesh2, point_cloud):
    """
    Computes the projection of points, e.g. a scan to a representative mesh. 
    Then carry over the points to a second representative mesh of a different shape and convert back to euclidean coordinates

    Parameters
    -------------------------------
    mesh1, mesh2: <class: trimesh.TriMesh> representative meshes of 2 shapes, where the vertices and edges of 
                  corresponding parts have the same indices
    point_cloud : (n, 3) coordinates of the points

    Output
    -------------------------------
    p           : (n, 3) projected coordinates of the points
    """

    # project points to first mesh
    b, tid = project_pc_to_triangles(mesh1.vertlist, mesh1.facelist, point_cloud, return_vec=True)

    # carry barycentric coord over to second mesh and calculate euclideaan coord from there
    p = mesh2.barycentric_to_points(b, tid)

    return p
