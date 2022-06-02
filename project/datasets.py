import os
from pathlib import Path
import itertools

import numpy as np
import potpourri3d as pp3d
import torch
from torch.utils.data import Dataset

from diffusion_net import geometry as dng
from diffusion_net import utils as dnu
from .utils import farthest_point_sample, square_distance
from . import projection_utils as pju 
from .trimesh import TriMeshPly as tm

class Dataset:

    def __init__(self, path, use_adj=False):
        #TODO: create check if its a pathlib.Path instance, otherwise create one
        self.path = path # path to dataset
        self.use_adj = use_adj

    # def get_samples(self):
    #     raise NotImplementedError

    def load_trimesh(self, idx):
        """
        Load TriMesh class for given ID.

        Parameters
        -------------------------------
        ids  : list list of ID's of the corresponding Shrec models

        Outputs
        -------------------------------
        meshes  : list of two meshes as trimesh.TriMesh class
        """
        idx1, idx2 = self.combinations[idx]
        ids = [idx1, idx2]
        meshes = [tm(self.verts_list[i], self.faces_list[i]) for i in ids]

        return meshes

    def get_p2p(self, idx):
        """
        Get point to point map for given ID.
        Parameters
        -------------------------------
        idx : ID of combination to get point to point map for
        
        Outputs
        -------------------------------
        p2p_map : point to point map
        """
        idx1, idx2 = self.combinations[idx]
        p2p_map = self.corres_dict[(idx1, idx2)]

        if p2p_map.size(0) == self.verts_list[idx1].size(0) + self.verts_list[idx2].size(0):
            p2p_map = p2p_map[:self.verts_list[idx2].size(0)]

        return p2p_map

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        idx1, idx2 = self.combinations[idx]

        shape1 = {
            "xyz": self.verts_list[idx1],
            "faces": self.faces_list[idx1]
        }

        shape2 = {
            "xyz": self.verts_list[idx2],
            "faces": self.faces_list[idx2]
        }

        # Compute fmap
        map21 = self.corres_dict[(idx1, idx2)]

        return {"shape1": shape1, "shape2": shape2, "map21": map21}
    

class Faust:
    # Calculating p2p ground truth maps for evaluation takes too long!
    # Use other methods
    def __init__(self, path):

        #TODO: create check if its a pathlib.Path instance, otherwise create one
        self.path = path # path to FAUST dataset
    

    def get_samples(self, sample_nr, test=False, scan=True):
        mode = "test" if test else "training"
        mode_prefix = "test_" if test else "tr_"
        type = "scans" if scan else "registrations"
        type_prefix = "scan_" if scan else "reg_"

        paths = [self.path / mode / type / (mode_prefix + type_prefix + nr + ".ply") for nr in sample_nr]

        meshes = [tm(str(path)) for path in paths]

        return meshes

    def calc_point_to_surface_map(self, shape1, shape2):
        """
        Calculate the point to surface map from shape1 to shape2

        Parameters
        -------------------------------
        shape1, shape2  : str three digit number corresponding to the sample number in the FAUST dataset

        Outputs
        -------------------------------
    
        """

        # Load the two representation meshes
        rep_mesh1, rep_mesh2 = self.get_samples([shape1, shape2], test=False, scan=False)

        print("got representation meshes")
        # Load the scan meshes
        # scan_mesh1, scan_mesh2 = self.get_samples([shape1, shape2], test=False, scan=True)
        scan_mesh1 = self.get_samples([shape1], test=False, scan=True)[0]
        print("Got first scan mesh")
        rep_mesh2_points = pju.project_point_to_rep_to_rep(rep_mesh1, rep_mesh2, scan_mesh1.vertlist)
        print("calculated first projection")
        del scan_mesh1
        scan_mesh2 = self.get_samples([shape2], test=False, scan=True)[0]

        b, tid = pju.project_pc_to_triangles(scan_mesh2.vertlist, scan_mesh2.facelist, rep_mesh2_points, return_vec=True)
        
        scan2_points = scan_mesh2.barycentric_to_points(b, tid)

        del scan_mesh2

        return scan2_points

    def get_geodesic_error(self):

        pass

class FaustRep:

    def __init__(self, path, name="reg", use_adj=False, verbose=True):
        self.path = path # path to FAUST dataset
        self.used_shapes = sorted([x.stem for x in (Path(path) / "training" / "registrations").iterdir() if (name in x.stem and ".ply" in str(x))])

        mesh_dirpath = Path(path) / "training" / "registrations"

        # Get all the files
        self.verts_list = []
        self.faces_list = []
        self.sample_list = []

        # Load the actual files
        for shape_name in self.used_shapes:
            # On Mac, iCloud creates .DS_Store files in a directory
            if ".DS_Store" in shape_name:
                continue
            if verbose:
                print("loading mesh " + str(shape_name))

            verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}.ply"))

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            self.verts_list.append(verts)
            self.faces_list.append(faces)
            # idx0 = farthest_point_sample(verts.t(), ratio=0.9)
            # dists, idx1 = square_distance(verts.unsqueeze(0), verts[idx0].unsqueeze(0)).sort(dim=-1)
            # dists, idx1 = dists[:, :, :130].clone(), idx1[:, :, :130].clone()
            # self.sample_list.append((idx0, idx1, dists))

        all_combs = [i for subset in itertools.combinations(self.used_shapes, 2) for i in (subset, tuple(reversed(subset)))]
        self.corres_dict = {}
        for x, y in all_combs:
            map_ = torch.tensor(range(len(self.verts_list[self.used_shapes.index(y)]))).long()
            self.corres_dict[(self.used_shapes.index(y), self.used_shapes.index(x))] = map_

        # set combinations
        self.combinations = list(self.corres_dict.keys())

        if verbose:
            print("Initialization done!")

    def load_trimesh(self, idx):
        """
        Load TriMesh class for given ID.

        Parameters
        -------------------------------
        ids  : list list of ID's of the corresponding Shrec models

        Outputs
        -------------------------------
        meshes  : list of two meshes as trimesh.TriMesh class
        """
        idx1, idx2 = self.combinations[idx]
        ids = [idx1, idx2]
        meshes = [tm(self.verts_list[i], self.faces_list[i]) for i in ids]

        return meshes

    def get_p2p(self, idx):
        """
        Get point to point map for given ID.
        Parameters
        -------------------------------
        idx : ID of combination to get point to point map for
        
        Outputs
        -------------------------------
        p2p_map : point to point map
        """
        idx1, idx2 = self.combinations[idx]
        p2p_map = self.corres_dict[(idx1, idx2)]

        if p2p_map.size(0) == self.verts_list[idx1].size(0) + self.verts_list[idx2].size(0):
            p2p_map = p2p_map[:self.verts_list[idx2].size(0)]

        return p2p_map

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        idx1, idx2 = self.combinations[idx]

        shape1 = {
            "xyz": self.verts_list[idx1],
            "faces": self.faces_list[idx1]
        }

        shape2 = {
            "xyz": self.verts_list[idx2],
            "faces": self.faces_list[idx2]
        }

        # Compute fmap
        map21 = self.corres_dict[(idx1, idx2)]

        # TODO: Add calculation of C_gt
        # evec_1, evec_2, mass2 = shape1["evecs"][:, :self.n_fmap], shape2["evecs"][:, :self.n_fmap], shape2["mass"]
        # trans_evec2 = evec_2.t() @ torch.diag(mass2)

        # P = torch.zeros(evec_2.size(0), evec_1.size(0))
        # P[range(evec_2.size(0)), map21.flatten()] = 1
        # C_gt = trans_evec2 @ P @ evec_1

        return {"shape1": shape1, "shape2": shape2, "map21": map21}


class ShrecPartialDataset(Dataset):
    def __init__(self, root_dir, name="cuts", k_eig=128, n_fmap=30, use_cache=True, op_cache_dir=None, use_adj=False, verbose=True):

        self.k_eig = k_eig
        self.n_fmap = n_fmap
        self.root_dir = root_dir
        self.cache_dir = root_dir
        self.op_cache_dir = op_cache_dir

        # check the cache
        if use_cache:
            load_cache = os.path.join(self.cache_dir, f"cache_{name}_train.pt")
            if verbose:
                print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                if verbose:
                    print("  --> loading dataset from cache")
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    self.used_shapes,
                    self.corres_dict,
                    self.sample_list,
                ) = torch.load(load_cache)
                self.combinations = list(self.corres_dict.keys())
                return
            if verbose:
                print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels
        # define files and order
        self.used_shapes = sorted([x.stem for x in (Path(root_dir) / "shapes").iterdir() if name in x.stem])
        corres_path = Path(root_dir) / "maps"
        all_combs = [x.stem for x in corres_path.iterdir() if name in x.stem]
        self.corres_dict = {}
        for x, y in map(lambda x: (x[:x.rfind("_")], x[x.rfind("_") + 1:]), all_combs):
            if x in self.used_shapes and y in self.used_shapes:
                map_ = torch.from_numpy(np.loadtxt(corres_path / f"{x}_{y}.map", dtype=np.int32)).long()
                self.corres_dict[(self.used_shapes.index(y), self.used_shapes.index(x))] = map_

        # set combinations
        self.combinations = list(self.corres_dict.keys())
        mesh_dirpath = Path(root_dir) / "shapes"

        # Get all the files
        self.verts_list = []
        self.faces_list = []
        self.sample_list = []

        # Load the actual files
        for shape_name in self.used_shapes:
            # On Mac, iCloud creates .DS_Store files in a directory
            if ".DS_Store" in shape_name:
                continue
            if verbose:
                print("loading mesh " + str(shape_name))

            verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}.off"))

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            self.verts_list.append(verts)
            self.faces_list.append(faces)
            idx0 = farthest_point_sample(verts.t(), ratio=0.9)
            dists, idx1 = square_distance(verts.unsqueeze(0), verts[idx0].unsqueeze(0)).sort(dim=-1)
            dists, idx1 = dists[:, :, :130].clone(), idx1[:, :, :130].clone()
            self.sample_list.append((idx0, idx1, dists))
        # Precompute operators
        (
            self.frames_list,
            self.massvec_list,
            self.L_list,
            self.evals_list,
            self.evecs_list,
            self.gradX_list,
            self.gradY_list,
        ) = dng.get_all_operators(
            self.verts_list,
            self.faces_list,
            k_eig=self.k_eig,
            op_cache_dir=self.op_cache_dir,
        )

        # save to cache
        if use_cache:
            dnu.ensure_dir_exists(self.cache_dir)
            torch.save(
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    self.used_shapes,
                    self.corres_dict,
                    self.sample_list,
                ),
                load_cache,
            )
        if verbose:
            print("Initialization done!")

    def load_trimesh(self, idx):
        """
        Load TriMesh class for given ID.

        Parameters
        -------------------------------
        ids  : list list of ID's of the corresponding Shrec models

        Outputs
        -------------------------------
        meshes  : list of two meshes as trimesh.TriMesh class
        """
        idx1, idx2 = self.combinations[idx]
        ids = [idx1, idx2]
        meshes = [tm(self.verts_list[i], self.faces_list[i]) for i in ids]

        return meshes

    def get_p2p(self, idx):
        """
        Get point to point map for given ID.
        Parameters
        -------------------------------
        idx : ID of combination to get point to point map for
        
        Outputs
        -------------------------------
        p2p_map : point to point map
        """
        idx1, idx2 = self.combinations[idx]
        p2p_map = self.corres_dict[(idx1, idx2)]

        if p2p_map.size(0) == self.verts_list[idx1].size(0) + self.verts_list[idx2].size(0):
            p2p_map = p2p_map[:self.verts_list[idx2].size(0)]
        

        return p2p_map

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        idx1, idx2 = self.combinations[idx]

        shape1 = {
            "xyz": self.verts_list[idx1],
            "faces": self.faces_list[idx1],
            "frames": self.frames_list[idx1],
            "mass": self.massvec_list[idx1],
            "L": self.L_list[idx1],
            "evals": self.evals_list[idx1],
            "evecs": self.evecs_list[idx1],
            "gradX": self.gradX_list[idx1],
            "gradY": self.gradY_list[idx1],
            "name": self.used_shapes[idx1],
            "sample_idx": self.sample_list[idx1],
        }

        shape2 = {
            "xyz": self.verts_list[idx2],
            "faces": self.faces_list[idx2],
            "frames": self.frames_list[idx2],
            "mass": self.massvec_list[idx2],
            "L": self.L_list[idx2],
            "evals": self.evals_list[idx2],
            "evecs": self.evecs_list[idx2],
            "gradX": self.gradX_list[idx2],
            "gradY": self.gradY_list[idx2],
            "name": self.used_shapes[idx2],
            "sample_idx": self.sample_list[idx2],
        }

        # Compute fmap
        map21 = self.corres_dict[(idx1, idx2)]

        evec_1, evec_2, mass2 = shape1["evecs"][:, :self.n_fmap], shape2["evecs"][:, :self.n_fmap], shape2["mass"]
        trans_evec2 = evec_2.t() @ torch.diag(mass2)

        # If the size of the map vector is too long, shorten it
        # the removed part only contains 0's and 1's
        # TODO: check if this should be handled differently and if it still gives a viable map!
        if map21.size(0) == evec_2.size(0) + evec_1.size(0):
            map21 = map21[:evec_2.size(0)]

        P = torch.zeros(evec_2.size(0), evec_1.size(0))
        P[range(evec_2.size(0)), map21.flatten()] = 1
        C_gt = trans_evec2 @ P @ evec_1

        # compute region labels
        gt_partiality_mask12 = torch.zeros(shape1["xyz"].size(0)).long().detach()
        gt_partiality_mask12[map21[map21 != -1]] = 1
        gt_partiality_mask21 = torch.zeros(shape2["xyz"].size(0)).long().detach()
        gt_partiality_mask21[map21 != -1] = 1

        return {"shape1": shape1, "shape2": shape2, "C_gt": C_gt,
                "map21": map21, "gt_partiality_mask12": gt_partiality_mask12, "gt_partiality_mask21": gt_partiality_mask21}

class Tosca(Dataset):
    """
    Download dataset from https://vision.in.tum.de/data/datasets/partial
    """
    def __init__(self, path, name="cuts", selected=False, use_adj=False, verbose=True):
        super().__init__(path, use_adj=use_adj)

        if selected:
            mesh_dirpath = Path(self.path) / "selected" / name
        else:
            mesh_dirpath = Path(self.path) / name
        null_shape_dirpath = Path(self.path) / "null"

        self.used_shapes = sorted([x.stem for x in (mesh_dirpath).iterdir() if ".off" in str(x)])
        self.null_shapes = sorted([x.stem for x in (null_shape_dirpath).iterdir() if ".off" in str(x)])

        # Get all the files
        self.verts_list = []
        self.faces_list = []
        self.sample_list = []

        # Load the actual files
        for shape_name in self.used_shapes:
            # On Mac, iCloud creates .DS_Store files in a directory
            if ".DS_Store" in shape_name:
                continue
            if verbose:
                print("loading mesh " + str(shape_name))

            verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}.off"))

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            self.verts_list.append(verts)
            self.faces_list.append(faces)
            # idx0 = farthest_point_sample(verts.t(), ratio=0.9)
            # dists, idx1 = square_distance(verts.unsqueeze(0), verts[idx0].unsqueeze(0)).sort(dim=-1)
            # dists, idx1 = dists[:, :, :130].clone(), idx1[:, :, :130].clone()
            # self.sample_list.append((idx0, idx1, dists))

        for shape_name in self.null_shapes:
            # On Mac, iCloud creates .DS_Store files in a directory
            if ".DS_Store" in shape_name:
                continue
            if verbose:
                print("loading mesh " + str(shape_name))

            verts, faces = pp3d.read_mesh(str(null_shape_dirpath / f"{shape_name}.off"))

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            self.verts_list.append(verts)
            self.faces_list.append(faces)
            # idx0 = farthest_point_sample(verts.t(), ratio=0.9)
            # dists, idx1 = square_distance(verts.unsqueeze(0), verts[idx0].unsqueeze(0)).sort(dim=-1)
            # dists, idx1 = dists[:, :, :130].clone(), idx1[:, :, :130].clone()
            # self.sample_list.append((idx0, idx1, dists))
        
        # Meshes are in the same folder as maps
        corres_path = mesh_dirpath
        # The correspondences are given wrt the null shape
        all_combs = [x.stem for x in corres_path.iterdir() if ".gt" in str(x)]
        self.corres_dict = {}
        if not self.use_adj:
            self.mask_dict = {}
        for x in all_combs:
            y = x.split("_")[0]
            if x in self.used_shapes and y in self.null_shapes:
                # map towards high-res original, indices are starting at 1, not 0
                map_x_hr = torch.from_numpy(np.loadtxt(corres_path / f"{x}.gt", dtype=np.int32)).long()
                map_x_hr = torch.tensor([x-1 for x in map_x_hr])
                # map from high-res original to remeshed
                map_hr_re = torch.from_numpy(np.loadtxt(null_shape_dirpath / f"{y}.remesh", dtype=np.int32)).long()
                map_hr_re = torch.tensor([x-1 for x in map_hr_re])
                # get map from partial shape to null shape
                gt_map12 = map_hr_re[map_x_hr]
                # get map from null to partial if use_adj == False, i.e. p2p map describes mesh2 -> mesh1
                if not self.use_adj:
                    # create list of zeros with length = number of vertices on null shape
                    gt_map21 = list([-1] * len(self.verts_list[(len(self.used_shapes) + self.null_shapes.index(y))]))
                    for j, i in enumerate(gt_map12):
                        gt_map21[i] = j
                    
                    # create mask for unmapped vertices in mesh2
                    mask_21 = [x for x in range(len(gt_map21)) if gt_map21[x] == -1]

                    self.corres_dict[(self.used_shapes.index(x), (len(self.used_shapes) + self.null_shapes.index(y)))] = gt_map21
                    self.masks_dict[(self.used_shapes.index(x), (len(self.used_shapes) + self.null_shapes.index(y)), )] = mask_21
                else:
                    self.corres_dict[((len(self.used_shapes) + self.null_shapes.index(y)), self.used_shapes.index(x))] = gt_map12

        # set combinations
        self.combinations = list(self.corres_dict.keys())

        if verbose:
            print("Initialization done!")


def shape_to_device(dict_shape, device):
    names_to_device = ["xyz", "faces", "mass", "evals", "evecs", "gradX", "gradY"]
    for k, v in dict_shape.items():
        if "shape" in k:
            for name in names_to_device:
                v[name] = v[name].to(device)
            dict_shape[k] = v
        else:
            dict_shape[k] = v.to(device)

    return dict_shape
