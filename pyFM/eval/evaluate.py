import numpy as np


def accuracy(p2p, gt_p2p, D1_geod, return_all=False, sqrt_area=None):
    """
    Computes the geodesic accuracy of a vertex to vertex map. The map goes from
    the target shape to the source shape.

    Parameters
    ----------------------
    p2p        : (n2,) - vertex to vertex map giving the index of the matched vertex on the source shape
                 for each vertex on the target shape (from a functional map point of view)
    gt_p2p     : (n2,) - ground truth mapping between the pairs
    D1_geod    : (n1,n1) - geodesic distance between pairs of vertices on the source mesh
    return_all : bool - whether to return all the distances or only the average geodesic distance

    Output
    -----------------------
    acc   : float - average accuracy of the vertex to vertex map
    dists : (n2,) - if return_all is True, returns all the pairwise distances
    """

    dists = D1_geod[(p2p,gt_p2p)]
    if sqrt_area is not None:
        dists /= sqrt_area

    if return_all:
        return dists.mean(), dists

    return dists.mean()


def continuity(p2p, D1_geod, D2_geod, edges):
    """
    Computes continuity of a vertex to vertex map. The map goes from
    the target shape to the source shape.

    Parameters
    ----------------------
    p2p     : (n2,) - vertex to vertex map giving the index of the matched vertex on the source shape
                 for each vertex on the target shape (from a functional map point of view)
    gt_p2p  : (n2,) - ground truth mapping between the pairs
    D1_geod : (n1,n1) - geodesic distance between pairs of vertices on the source mesh
    D2_geod : (n1,n1) - geodesic distance between pairs of vertices on the target mesh
    edges   : (n2,2) edges on the target shape

    Output
    -----------------------
    continuity : float - average continuity of the vertex to vertex map
    """
    source_len = D2_geod[(edges[:,0], edges[:,1])]
    target_len = D1_geod[(p2p[edges[:,0]], p2p[edges[:,1]])]

    continuity = np.mean(target_len / source_len)

    return continuity


def coverage(p2p, A):
    """
    Computes coverage of a vertex to vertex map. The map goes from
    the target shape to the source shape.

    Parameters
    ----------------------
    p2p : (n2,) - vertex to vertex map giving the index of the matched vertex on the source shape
                 for each vertex on the target shape (from a functional map point of view)
    A   : (n1,n1) or (n1,) - area matrix on the source shape or array of per-vertex areas.

    Output
    -----------------------
    coverage : float - coverage of the vertex to vertex map
    """
    if len(A.shape) == 2:
        vert_area = np.asarray(A.sum(1)).flatten()
    coverage = vert_area[np.unique(p2p)].sum() / vert_area.sum()

    return coverage

class EvaluateModel:

    def __init__(self, model, data_class, data_path, refine=False, preprocess_params={}, fit_params={}, verbose=False):
        """
        Initialize Class

        Parameters
        -----------------------
        model       : Class for model, should contain functions preprocess, fit and 
                        if refine==True, then also icp_refine and zoomout_refine.
        data_class  : Initialized data class, should contain __getitem__ accessing all possible combinations where ground thruth maps are available,
                        __len__, get_p2p and load_trimesh function.
        """

        self.model = model
        self.refine = refine
        self.preprocess_params = preprocess_params
        self.fit_params = fit_params
        self.verbose = verbose

        self.acc_base = []
        if verbose:
            self.acc_icp = []
            self.acc_zo = []

        # TODO: remove name=""
        self.data = data_class(data_path, name="")

    def eval(self):
        """
        Evaluate given method on given dataset. For now only supports canonical accuracy implemented in pyFM.
        """
        for idx in range(len(self.data)):
            # TODO: consider precomputing evecs, etc. for each mesh to avoid multiple computations for the same mesh
            mesh1, mesh2 = self.data.load_trimesh(idx)
            model = self.model(mesh1, mesh2)

            model.preprocess(**self.preprocess_params, verbose=self.verbose)
            model.fit(**self.fit_params, verbose=self.verbose)
            p2p_21 = model.get_p2p()

            if self.refine:
                model.icp_refine('classic', verbose=self.verbose)
                p2p_21_icp = model.get_p2p()
                model.zoomout_refine('classic', verbose=self.verbose)
                p2p_21_zo = model.get_p2p()

            A_geod = mesh1.get_geodesic(verbose=self.verbose)
            gt_p2p = self.data.get_p2p(idx)

            self.acc_base.append(accuracy(p2p_21, gt_p2p, A_geod, sqrt_area=mesh1.sqrtarea))

            if self.refine:
                self.acc_icp.append(accuracy(p2p_21_icp, gt_p2p, A_geod, sqrt_area=np.sqrt(mesh1.area)))
                self.acc_zo.append(accuracy(p2p_21_zo, gt_p2p, A_geod, sqrt_area=np.sqrt(mesh1.area)))

        if self.refine:
            print(f'Mean accuracy results\n'
                f'\tBasic FM : {1e3*np.array(self.acc_base).mean():.2f}\n'
                f'\tICP refined : {1e3*np.array(self.acc_icp).mean():.2f}\n'
                f'\tZoomOut refined : {1e3*np.array(self.acc_zo).mean():.2f}\n')

            print(f'Standard deviation of accuracy results\n'
                f'\tBasic FM : {1e3*np.array(self.acc_base).std():.2f}\n'
                f'\tICP refined : {1e3*np.array(self.acc_icp).std():.2f}\n'
                f'\tZoomOut refined : {1e3*np.array(self.acc_zo).std():.2f}\n')
        else:
            print(f'Mean accuracy results\n'
                f'\tBasic FM : {1e3*np.array(self.acc_base).mean():.2f}\n')

            print(f'Standard deviation of accuracy results\n'
                f'\tBasic FM : {1e3*np.array(self.acc_base).std():.2f}\n')

        with open("data/eval_results.txt", "w+") as f:
            self.str_acc_base = [str(x) for x in self.acc_base]
            f.write(", ".join(self.str_acc_base))

            if self.refine:
                self.str_acc_icp = [str(x) for x in self.acc_icp]
                self.str_acc_zo = [str(x) for x in self.acc_zo]
                f.write(", ".join(self.str_acc_icp))
                f.write(", ".join(self.str_acc_zo))
