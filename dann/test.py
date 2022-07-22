import torch
import numpy as np
import tqdm
import torch.optim as optim
import torch.nn as nn
import test
import os

from torch.utils.tensorboard import SummaryWriter

from dann.utils import save_model, set_model_mode, DPFMLoss_da

from dann.model import FeatExtractorNet, MapExtractorNet, Discriminator, DPFMNet_DA
#from utils import visualize
from project.datasets import ShrecPartialDataset, Tosca, shape_to_device
from dpfm.utils import augment_batch

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

    return p2p_pred, overlap_score12, overlap_score21, use_feat1, use_feat2, log_obj


def test_target(cfg, target_test_loader, model_path, save_name, predictions_name):
    
    print(f"Starting evaluation...")
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")
    
    
    to_save_list = []
    acc_list = []
    pred_p2p_list = []
    distances = []
    # define model
    print(f"Loading model...")
    dpfm_da_net = DPFMNet_DA(cfg).to(device)
    dpfm_da_net.load_state_dict(torch.load(model_path, map_location=device))
    dpfm_da_net.eval()

    print(f"Starting inference...")
    for i, data in enumerate(tqdm.tqdm(target_test_loader)):

        data = shape_to_device(data, device)

        # prepare iteration data
        shape1, shape2, p2p_gt = data["shape1"], data["shape2"], data["map21"]

        # do iteration
        p2p_pred, _, _, _, _, _ = FM_batch_eval(data, dpfm_da_net, shape1, shape2)
        
       
        pred_p2p_list.append({'p2p': p2p_pred, 'mesh1': shape1["mesh"], 'mesh2': shape2["mesh"]})#, 'overlap_12': overlap_score12, 'overlap_21': overlap_score21, 'feat1': use_feat1, 'feat2': use_feat2})

        to_save_list.append(log_obj)

        mesh1_geod = shape1["mesh"].get_geodesic()
        mesh1_sqrt_area = shape1["mesh"].sqrtarea

        

        mean_dist, dist = accuracy(p2p_pred, p2p_gt.cpu(), mesh1_geod, sqrt_area=mesh1_sqrt_area, return_all=True)
        distances.extend(dist)
        acc_list.append(mean_dist)
        
    print(f"Mean normalized geodesic error: {sum(acc_list)/len(acc_list)}")
    torch.save(to_save_list, predictions_name)

    return pred_p2p_list, distances

def test_source(cfg, source_test_loadet, model_path, save_name, predictions_name):
    
    print(f"Starting evaluation...")
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")
    
    
    to_save_list = []
    acc_list = []
    pred_p2p_list = []
    distances = []
    # define model
    print(f"Loading model...")
    dpfm_da_net = DPFMNet_DA(cfg).to(device)
    dpfm_da_net.load_state_dict(torch.load(model_path, map_location=device))
    dpfm_da_net.eval()

    print(f"Starting inference...")
    for i, data in enumerate(tqdm.tqdm(source_test_loader)):

        data = shape_to_device(data, device)

        # prepare iteration data
        shape1, shape2, p2p_gt = data["shape1"], data["shape2"], data["map21"]

        # do iteration
        p2p_pred, _, _, _, _, _ = FM_batch_eval(data, dpfm_da_net, shape1, shape2)
        
       
        pred_p2p_list.append({'p2p': p2p_pred, 'mesh1': shape1["mesh"], 'mesh2': shape2["mesh"]})#, 'overlap_12': overlap_score12, 'overlap_21': overlap_score21, 'feat1': use_feat1, 'feat2': use_feat2})

        to_save_list.append(log_obj)

        mesh1_geod = shape1["mesh"].get_geodesic()
        mesh1_sqrt_area = shape1["mesh"].sqrtarea

        

        mean_dist, dist = accuracy(p2p_pred, p2p_gt.cpu(), mesh1_geod, sqrt_area=mesh1_sqrt_area, return_all=True)
        distances.extend(dist)
        acc_list.append(mean_dist)
        
    print(f"Mean normalized geodesic error: {sum(acc_list)/len(acc_list)}")
    torch.save(to_save_list, predictions_name)

    return pred_p2p_list, distances    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    