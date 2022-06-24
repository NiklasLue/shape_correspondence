import argparse
from pickle import TRUE
import yaml
import os
import statistics

import torch
import tqdm

from project.datasets import ShrecPartialDataset, Tosca, shape_to_device
from dpfm.model import DPFMNet
from pyFM.spectral import FM_to_p2p
from pyFM.eval.evaluate import accuracy
from project.cfunctional import CoupledFunctionalMapping


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

def CFM_batch_eval(batch_data, net, shape1, shape2):
    """
    Function evaluating a batch of shapes with the Coupled Functional Map Framework
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
    
    print("Initializing CFM framework...")
    cfm = CoupledFunctionalMapping(shape1["mesh"], shape2["mesh"])
    print("Start calculation of CFM")
    C1, C2 = cfm.fit_pre_comp(use_feat1.detach().numpy().squeeze(0), use_feat2.detach().numpy().squeeze(0), shape1["evecs"].detach().numpy(), shape2["evecs"].detach().numpy(), shape1["evals"].detach().numpy(), shape2["evals"].detach().numpy(), verbose=True)
    print("calc FM done")
    p2p_pred = FM_to_p2p(C1.detach().cpu().squeeze(0), shape1["evecs"][:, :k1].cpu(), shape2["evecs"][:, :k2].cpu(), use_adj=False, use_ANN=False, n_jobs=1)
    print("calc p2p done")

    # create object to append to to_save_list
    name1, name2 = batch_data["shape1"]["name"], batch_data["shape2"]["name"]
    log_obj = (name1, name2, C1.detach().cpu().squeeze(0), C_gt.detach().cpu().squeeze(0),
                            gt_partiality_mask12.detach().cpu().squeeze(0), gt_partiality_mask21.detach().cpu().squeeze(0),
                            p2p_pred, batch_data["map21"])

    return p2p_pred, log_obj


def eval_net(cfg, model_path, predictions_name, mode="FM"):
    """
    Rewritten eval_net() function from DPFM
    """
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = cfg["dataset"]["cache_dir"]
    dataset_path = cfg["dataset"]["root_test"]

    # create dataset
    if cfg["dataset"]["name"] == "shrec16":
        test_dataset = ShrecPartialDataset(dataset_path, name=cfg["dataset"]["subset"], k_eig=cfg["fmap"]["k_eig"],
                                           n_fmap=cfg["fmap"]["n_fmap"], use_cache=True, op_cache_dir=op_cache_dir)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, shuffle=True)
    elif cfg["dataset"]["name"] == "tosca":
        # TODO: adjust use_adj, so that it matches with pyFM
        test_dataset = Tosca(dataset_path, name=cfg["dataset"]["subset"], k_eig=cfg["fmap"]["k_eig"],
                                           n_fmap=cfg["fmap"]["n_fmap"], use_cache=True, op_cache_dir=op_cache_dir, use_adj=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None)
    elif cfg["dataset"]["name"] == "faust":
        raise NotImplementedError("FAUST support will come soon!")
    else:
        raise NotImplementedError("dataset not implemented!")

    # define model
    dpfm_net = DPFMNet(cfg).to(device)
    dpfm_net.load_state_dict(torch.load(model_path, map_location=device))
    dpfm_net.eval()

    # initialize logging lists
    to_save_list = []
    acc_list = []

    # enumerate through batches of data
    for i, data in enumerate(tqdm.tqdm(test_loader)):

        data = shape_to_device(data, device)

        # prepare iteration data
        shape1, shape2, p2p_gt = data["shape1"], data["shape2"], data["map21"]
        mesh1, mesh2 = shape1["mesh"], shape2["mesh"]

        # do iteration
        if mode=="FM":
            p2p_pred, log_obj = FM_batch_eval(data, dpfm_net, shape1, shape2)
            return p2p_pred, data, log_obj
        elif mode=="CFM":
            p2p_pred, log_obj = CFM_batch_eval(data, dpfm_net, shape1, shape2)
        else:
            message = f"mode '{mode}' not supported!"
            raise ValueError(message)

        # log results
        to_save_list.append(log_obj)
        acc_list.append(accuracy(p2p_pred, p2p_gt.cpu(), mesh1.get_geodesic(), sqrt_area=mesh1.sqrtarea))

    print(f"Mean normalized geodesic error: {statistics.fmean(acc_list)}")
    torch.save(to_save_list, predictions_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of DPFM model.")

    parser.add_argument("--config", type=str, default="tosca_cuts", help="Config file name")

    parser.add_argument("--model_path", type=str, help="path to saved model")
    parser.add_argument("--predictions_name", type=str, help="name of the prediction file")
    parser.add_argument("--mode", type=str, default="CFM", help="FM or CFM")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"../dpfm/config/{args.config}.yaml", "r"))
    eval_net(cfg, args.model_path, args.predictions_name, mode=args.mode)
