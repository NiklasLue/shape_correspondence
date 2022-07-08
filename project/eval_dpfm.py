import argparse
from pickle import TRUE
import yaml
import os

import torch
import tqdm

from project.datasets import ShrecPartialDataset, Tosca, shape_to_device
from dpfm.model import DPFMNet
from pyFM.refine import icp
from pyFM.spectral import FM_to_p2p
from pyFM.eval.evaluate import accuracy
from project.cfunctional import CoupledFunctionalMapping
from project.cfunctional_dpfm import CoupledFunctionalMappingDPFM


def FM_batch_eval(batch_data, net, shape1, shape2):
    """
    Function evaluating a batch of shapes with the Functional Map Framework
    batch_data: A batch of data from a PyTorch dataloader
    net: instance of DPFMNet class
    shape1, shape2: shape objects given by a data class in datasets.py
    """
    # initialized variables
    C_gt = batch_data["C_gt"].unsqueeze(0)
    gt_partiality_mask12, gt_partiality_mask21 = batch_data["gt_partiality_mask12"], batch_data["gt_partiality_mask21"]

    # calculate predicted FM and P2P map
    C_pred, overlap_score12, overlap_score21, use_feat1, use_feat2 = net(batch_data)
    _, k1, k2 = C_pred.shape
    p2p_pred = FM_to_p2p(C_pred.detach().cpu().squeeze(0), shape1["evecs"][:, :k1].cpu(), shape2["evecs"][:, :k2].cpu(), use_adj=False, use_ANN=False, n_jobs=1)
    
    # create object to append to to_save_list
    name1, name2 = batch_data["shape1"]["name"], batch_data["shape2"]["name"]
    log_obj = (name1, name2, C_pred.detach().cpu().squeeze(0), C_gt.detach().cpu().squeeze(0),
                            gt_partiality_mask12.detach().cpu().squeeze(0), gt_partiality_mask21.detach().cpu().squeeze(0),
                            p2p_pred, batch_data["map21"])

    return p2p_pred, log_obj


def eval_net(cfg, model_path, predictions_name, return_pred_p2p=False, return_dist=False, mode="FM", n_samples=None):
    """
    Rewritten eval_net() function from DPFM
    """
    #TODO: tidy up creation of output, depending on the return argumentss
    #TODO: create option not to calculate any geodesic distances, as this calculation takes the most amount of time
    print(f"Starting evaluation...")
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = cfg["dataset"]["cache_dir"]
    dataset_path = cfg["dataset"]["root_test"]

    # create dataset
    print(f"Loading data...")
    if cfg["dataset"]["name"] == "shrec16":
        test_dataset = ShrecPartialDataset(dataset_path, name=cfg["dataset"]["subset"], k_eig=cfg["fmap"]["k_eig"],
                                           n_fmap=cfg["fmap"]["n_fmap"], use_cache=True, op_cache_dir=op_cache_dir)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, shuffle=False)
    elif cfg["dataset"]["name"] == "tosca":
        # TODO: adjust use_adj, so that it matches with pyFM
        test_dataset = Tosca(dataset_path, name=cfg["dataset"]["subset"], k_eig=cfg["fmap"]["k_eig"],
                                           n_fmap=cfg["fmap"]["n_fmap"], use_cache=True, op_cache_dir=op_cache_dir, use_adj=True, n_samples=n_samples)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None)
    elif cfg["dataset"]["name"] == "faust":
        raise NotImplementedError("FAUST support will come soon!")
    else:
        raise NotImplementedError("dataset not implemented!")

    # define model
    print(f"Loading model...")
    dpfm_net = DPFMNet(cfg).to(device)
    dpfm_net.load_state_dict(torch.load(model_path, map_location=device))
    dpfm_net.eval()

    if mode=="CFM":
        cfm = CoupledFunctionalMappingDPFM(dpfm_net)

    # initialize logging lists
    to_save_list = []
    acc_list = []
    if return_pred_p2p:
        pred_p2p_list = []
    if return_dist:
        distances = []

    # enumerate through batches of data
    print(f"Starting inference...")
    for i, data in enumerate(tqdm.tqdm(test_loader)):

        data = shape_to_device(data, device)

        # prepare iteration data
        shape1, shape2, p2p_gt = data["shape1"], data["shape2"], data["map21"]

        # do iteration
        if mode=="FM":
            p2p_pred, log_obj = FM_batch_eval(data, dpfm_net, shape1, shape2)
        elif mode=="CFM":
            C1, C2 = cfm.fit(shape1, shape2, **cfg["cfm_eval"])
            p2p_pred = cfm.get_p2p_map(n_jobs=-1)
            log_obj = (data["shape1"]["name"], data["shape2"]["name"], 
                        C1, data["C_gt"].detach().cpu().unsqueeze(0),
                        data["gt_partiality_mask12"].detach().cpu().squeeze(0), data["gt_partiality_mask21"].detach().cpu().squeeze(0),
                        p2p_pred, data["map21"])
        else:
            message = f"mode '{mode}' not supported!"
            raise ValueError(message)

        if return_pred_p2p:
            pred_p2p_list.append({'p2p': p2p_pred, 'mesh1': shape1["mesh"], 'mesh2': shape2["mesh"]})

        to_save_list.append(log_obj)

        mesh1_geod = shape1["mesh"].get_geodesic()
        mesh1_sqrt_area = shape1["mesh"].sqrtarea

        if return_dist:

            mean_dist, dist = accuracy(p2p_pred, p2p_gt.cpu(), mesh1_geod, sqrt_area=mesh1_sqrt_area, return_all=True)
            distances.extend(dist)
            acc_list.append(mean_dist)
        else:    
            acc_list.append(accuracy(p2p_pred, p2p_gt.cpu(), mesh1_geod, sqrt_area=mesh1_sqrt_area))
        
    print(f"Mean normalized geodesic error: {sum(acc_list)/len(acc_list)}")
    torch.save(to_save_list, predictions_name)

    if return_pred_p2p and return_dist:
        return pred_p2p_list, distances
    elif return_pred_p2p:
        return pred_p2p_list
    elif return_dist:
        return distances

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of DPFM model.")

    parser.add_argument("--config", type=str, default="tosca_cuts", help="Config file name")

    parser.add_argument("--model_path", type=str, help="path to saved model")
    parser.add_argument("--predictions_name", type=str, help="name of the prediction file")
    parser.add_argument("--mode", type=str, default="CFM", help="FM or CFM")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"../dpfm/config/{args.config}.yaml", "r"))
    eval_net(cfg, args.model_path, args.predictions_name, mode=args.mode)
