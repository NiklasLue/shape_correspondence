import argparse
from pickle import TRUE
import yaml
import os

import torch
import tqdm

from project.datasets import ShrecPartialDataset, Tosca, shape_to_device
from dpfm.model import DPFMNet
from project.model_unsup import DPFMNet_unsup
from pyFM.spectral import FM_to_p2p
from pyFM.eval.evaluate import accuracy


def eval_net(cfg, model_path, predictions_name):
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
        test_dataset = Tosca(dataset_path, name=cfg["dataset"]["subset"], k_eig=cfg["fmap"]["k_eig"],
                                           n_fmap=cfg["fmap"]["n_fmap"], use_cache=True, op_cache_dir=op_cache_dir, use_adj=TRUE)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None)
    elif cfg["dataset"]["name"] == "faust":
        raise NotImplementedError("FAUST support will come soon!")
    else:
        raise NotImplementedError("dataset not implemented!")

    # define model
    dpfm_net = DPFMNet(cfg).to(device)
    dpfm_net.load_state_dict(torch.load(model_path, map_location=device))
    dpfm_net.eval()

    to_save_list = []
    acc_list = []
    for i, data in enumerate(tqdm.tqdm(test_loader)):

        data = shape_to_device(data, device)

        # prepare iteration data
        C_gt = data["C_gt"].unsqueeze(0)
        gt_partiality_mask12, gt_partiality_mask21 = data["gt_partiality_mask12"], data["gt_partiality_mask21"]
        shape1, shape2, p2p_gt = data["shape1"], data["shape2"], data["map21"]
        mesh1, mesh2 = shape1["mesh"], shape2["mesh"]

        # do iteration
        C_pred, overlap_score12, overlap_score21, use_feat1, use_feat2 = dpfm_net(data)
        _, k1, k2 = C_pred.shape
        p2p_pred = FM_to_p2p(C_pred.detach().cpu().squeeze(0), shape1["evecs"][:, :k1].cpu(), shape2["evecs"][:, :k2].cpu(), use_adj=True, use_ANN=False, n_jobs=1)

        name1, name2 = data["shape1"]["name"], data["shape2"]["name"]
        to_save_list.append((name1, name2, C_pred.detach().cpu().squeeze(0), C_gt.detach().cpu().squeeze(0),
                             gt_partiality_mask12.detach().cpu().squeeze(0), gt_partiality_mask21.detach().cpu().squeeze(0),
                             p2p_pred, p2p_gt))

        acc_list.append(accuracy(p2p_pred, p2p_gt.cpu(), mesh1.get_geodesic(), sqrt_area=mesh1.sqrtarea))

    print(f"Mean normalized geodesic error: {sum(acc_list)/len(acc_list)}")
    torch.save(to_save_list, predictions_name)
    
def eval_net_unsup(cfg, model_path, predictions_name):
    """
    eval_net() function from above adapted to setting of unsupervised model
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
        test_dataset = Tosca(dataset_path, name=cfg["dataset"]["subset"], k_eig=cfg["fmap"]["k_eig"],
                                           n_fmap=cfg["fmap"]["n_fmap"], use_cache=True, op_cache_dir=op_cache_dir, use_adj=TRUE)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None)
    elif cfg["dataset"]["name"] == "faust":
        raise NotImplementedError("FAUST support will come soon!")
    else:
        raise NotImplementedError("dataset not implemented!")

    # define model
    dpfm_net = DPFMNet_unsup(cfg).to(device)
    dpfm_net.load_state_dict(torch.load(model_path, map_location=device))
    dpfm_net.eval()

    to_save_list = []
    acc_list = []
    for i, data in enumerate(tqdm.tqdm(test_loader)):

        data = shape_to_device(data, device)

        # prepare iteration data
        C_gt = data["C_gt"].unsqueeze(0)
        gt_partiality_mask12, gt_partiality_mask21 = data["gt_partiality_mask12"], data["gt_partiality_mask21"]
        shape1, shape2, p2p_gt = data["shape1"], data["shape2"], data["map21"]
        mesh1, mesh2 = shape1["mesh"], shape2["mesh"]

        # do iteration
        C1_pred, C2_pred, use_feat1, use_feat2 = dpfm_net(data)
        _, k2, k1 = C1_pred.shape
        p2p_pred = FM_to_p2p(C1_pred.detach().cpu().squeeze(0), shape1["evecs"][:, :k1].cpu(), shape2["evecs"][:, :k2].cpu(), use_adj=True, use_ANN=False, n_jobs=1)

        name1, name2 = data["shape1"]["name"], data["shape2"]["name"]
        eval1, eval2 = data["shape1"]["evals"][:k1], data["shape2"]["evals"][:k2]
        to_save_list.append((name1, name2, C1_pred.detach().cpu().squeeze(0), C2_pred.detach().cpu().squeeze(0),
                             C_gt.detach().cpu().squeeze(0), gt_partiality_mask12.detach().cpu().squeeze(0),
                             gt_partiality_mask21.detach().cpu().squeeze(0), p2p_pred, p2p_gt, eval1, eval2))

        acc_list.append(accuracy(p2p_pred, p2p_gt.cpu(), mesh1.get_geodesic(), sqrt_area=mesh1.sqrtarea))

    print(f"Mean normalized geodesic error: {sum(acc_list)/len(acc_list)}")
    torch.save(to_save_list, predictions_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of DPFM model.")

    parser.add_argument("--config", type=str, default="shrec16_cuts", help="Config file name")

    parser.add_argument("--model_path", type=str, help="path to saved model")
    parser.add_argument("--predictions_name", type=str, help="name of the prediction file")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"../dpfm/config/{args.config}.yaml", "r"))
    eval_net(cfg, args.model_path, args.predictions_name)
