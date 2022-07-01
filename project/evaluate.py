import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pyFM.eval.evaluate import accuracy

class EvaluateModel:

    def __init__(self, model, data_class, data_path, refine=False, preprocess_params={}, fit_params={}, data_params={'name': ""}, verbose=False):
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

        self.data = data_class(data_path, **data_params)
        self.data_params = data_params

    def eval(self, accum_dist=True):
        """
        Evaluate given method on given dataset. For now only supports canonical accuracy implemented in pyFM.
        """
        if accum_dist:
            dist_list = []

        # Which p2p map to use. use_adj == True -> map12, use_adj == False -> map21
        if 'use_adj' in self.data_params:
            use_adj = self.data_params['use_adj']
        else:
            use_adj = False

        for idx in tqdm(range(len(self.data))):
            # TODO: consider precomputing evecs, etc. for each mesh to avoid multiple computations for the same mesh
            mesh1, mesh2 = self.data.load_trimesh(idx)
            model = self.model(mesh1, mesh2)

            model.preprocess(**self.preprocess_params, verbose=self.verbose)
            model.fit(**self.fit_params, verbose=self.verbose)
            p2p_21 = model.get_p2p(use_adj=use_adj)

            if self.refine:
                model.icp_refine('classic', verbose=self.verbose)
                p2p_21_icp = model.get_p2p(use_adj=use_adj)
                model.zoomout_refine('classic', verbose=self.verbose)
                p2p_21_zo = model.get_p2p(use_adj=use_adj)

            A_geod = mesh1.get_geodesic(verbose=self.verbose)
            gt_p2p = self.data.get_p2p(idx)

            if accum_dist:
                acc, dist = accuracy(p2p_21, gt_p2p, A_geod, sqrt_area=mesh1.sqrtarea, return_all=True)
                self.acc_base.append(acc)
                dist_list = np.concatenate((dist_list, dist), axis=None)
            else:
                self.acc_base.append(accuracy(p2p_21, gt_p2p, A_geod, sqrt_area=mesh1.sqrtarea))

            if self.refine:
                self.acc_icp.append(accuracy(p2p_21_icp, gt_p2p, A_geod, sqrt_area=np.sqrt(mesh1.area), return_all=accum_dist))
                self.acc_zo.append(accuracy(p2p_21_zo, gt_p2p, A_geod, sqrt_area=np.sqrt(mesh1.area), return_all=accum_dist))

        if self.refine:
            print(f'Mean accuracy results (*100)\n'
                f'\tBasic FM : {1e2*np.array(self.acc_base).mean():.2f}\n'
                f'\tICP refined : {1e2*np.array(self.acc_icp).mean():.2f}\n'
                f'\tZoomOut refined : {1e2*np.array(self.acc_zo).mean():.2f}\n')

            print(f'Standard deviation of accuracy results (*100)\n'
                f'\tBasic FM : {1e2*np.array(self.acc_base).std():.2f}\n'
                f'\tICP refined : {1e2*np.array(self.acc_icp).std():.2f}\n'
                f'\tZoomOut refined : {1e2*np.array(self.acc_zo).std():.2f}\n')
        else:
            print(f'Mean accuracy results (*100)\n'
                f'\tBasic FM : {1e2*np.array(self.acc_base).mean():.2f}\n')

            print(f'Standard deviation of accuracy results (*100)\n'
                f'\tBasic FM : {1e2*np.array(self.acc_base).std():.2f}\n')

        if accum_dist:
            X2 = np.sort(dist_list)
            F2 = np.array(range(len(dist_list)))/float(len(dist_list))

            plt.plot(X2, F2)
            plt.title('Cumulative geodesic error')
            plt.xlabel('Geodesic error')
            plt.ylabel('% Correspondences')
            # plt.xlim([0.0, 0.4])
            plt.ylim([0.0, 1.0])

            plt.savefig(f"data/eval/accum_dist_{type(model).__name__}_{type(self.data).__name__}.png", bbox_inches='tight', dpi=200)

            plt.close()

            with open(f"data/eval/distances_{type(model).__name__}_{type(self.data).__name__}.txt", "w+") as f:
                f.write(", ".join([str(x) for x in dist_list]))


        with open(f"data/eval/eval_results_{type(model).__name__}_{type(self.data).__name__}.txt", "w+") as f:
            self.str_acc_base = [str(x) for x in self.acc_base]
            f.write(", ".join(self.str_acc_base))

            if self.refine:
                self.str_acc_icp = [str(x) for x in self.acc_icp]
                self.str_acc_zo = [str(x) for x in self.acc_zo]
                f.write(", ".join(self.str_acc_icp))
                f.write(", ".join(self.str_acc_zo))
