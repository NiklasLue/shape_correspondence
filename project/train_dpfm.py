import os
import argparse
import yaml

import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from dpfm.model import DPFMNet
from project.model_unsup import DPFMNet_unsup
from dpfm.utils import DPFMLoss, augment_batch
from project.datasets import ShrecPartialDataset, Tosca, shape_to_device
from project.utils import DPFMLoss_unsup

def train_net(cfg, n_samples=None):
    """
    Rewritten train_net() function from DPFM to take more data classes
    """
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = cfg["dataset"]["cache_dir"]
    dataset_path = cfg["dataset"]["root_train"]

    save_dir_name = f'saved_models_{cfg["dataset"]["subset"]}_{cfg["dataset"]["model_name"]}'
    model_save_path = os.path.join(base_path, f"data/{save_dir_name}/ep" + "_{}.pth")
    if not os.path.exists(os.path.join(base_path, f"data/{save_dir_name}/")):
        os.makedirs(os.path.join(base_path, f"data/{save_dir_name}/"))

    # create dataset
    #TODO: implement training / validation split 
    if cfg["dataset"]["name"] == "shrec16":
        train_dataset = ShrecPartialDataset(dataset_path, name=cfg["dataset"]["subset"], k_eig=cfg["fmap"]["k_eig"],
                                            n_fmap=cfg["fmap"]["n_fmap"], use_cache=True, op_cache_dir=op_cache_dir)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=True)
    elif cfg["dataset"]["name"] == "tosca":
        train_dataset = Tosca(dataset_path, name=cfg["dataset"]["subset"], k_eig=cfg["fmap"]["k_eig"],
                                            n_fmap=cfg["fmap"]["n_fmap"], n_samples=n_samples, use_cache=True, op_cache_dir=op_cache_dir, use_adj=True)
        
        #TODO set train and validation ratio
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train, val = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        
        #Can increase num_workers to speed up training
        train_loader = torch.utils.data.DataLoader(train, batch_size=None, shuffle=True, num_workers=0)
        
        valid_loader = torch.utils.data.DataLoader(val, batch_size=None, shuffle=False, num_workers=0)
    elif cfg["dataset"]["name"] == "faust":
        raise NotImplementedError("FAUST support will come soon!")
    else:
        raise NotImplementedError("dataset not implemented!")

    # define model
    dpfm_net = DPFMNet(cfg).to(device)

    # if we are given a pretrained model, load the feature refiner and overlap predictor
    # NOTE: quick fix since we are not able to train the overlap predictor with the given code
    if "pretrained_state_dict" in cfg.keys():
        pretrained_sd = torch.load(f"{cfg['pretrained_state_dict']}", map_location=device) 
        model_dict = dpfm_net.state_dict()
        # 1. filter out unnecessary keys
        pretrained_sd = {k: v for k, v in pretrained_sd.items() if k in model_dict and k.split(".")[0] == "feat_refiner"}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_sd) 
        # 3. load the new state dict
        dpfm_net.load_state_dict(model_dict)

    lr = float(cfg["optimizer"]["lr"])
    optimizer = torch.optim.Adam(dpfm_net.parameters(), lr=lr, betas=(cfg["optimizer"]["b1"], cfg["optimizer"]["b2"]))
    criterion = DPFMLoss(w_fmap=cfg["loss"]["w_fmap"], w_acc=cfg["loss"]["w_acc"], w_nce=cfg["loss"]["w_nce"],
                         nce_t=cfg["loss"]["nce_t"], nce_num_pairs=cfg["loss"]["nce_num_pairs"]).to(device)

    # Training loop
    print("start training")
    iterations = 0
    lrs = []
    lambda1 = lambda epoch: 0.65 ** epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, verbose=True)
    writer = SummaryWriter()

    for epoch in tqdm.tqdm(range(1, cfg["training"]["epochs"] + 1)):
        dpfm_net.train()
    
        ### training step
        # train_loss = 0.0
        train_loss = []
        fmap_loss = []
        overlap_loss = []
        nce_loss = []
        for i, data in enumerate(train_loader):
            data = shape_to_device(data, device)

            # data augmentation
            data = augment_batch(data, rot_x=30, rot_y=30, rot_z=60, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)

            # prepare iteration data
            C_gt = data["C_gt"].unsqueeze(0)
            map21 = data["map21"]
            gt_partiality_mask12, gt_partiality_mask21 = data["gt_partiality_mask12"], data["gt_partiality_mask21"]

            # do iteration
            C_pred, overlap_score12, overlap_score21, use_feat1, use_feat2 = dpfm_net(data)
            out, fmap, overlap, nce = criterion(C_gt, C_pred, map21, use_feat1, use_feat2,
                             overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21)
            
            out.backward()
            torch.nn.utils.clip_grad_norm_(dpfm_net.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(out.item())
            fmap_loss.append(fmap.item())
            overlap_loss.append(overlap.item())
            nce_loss.append(nce.item())
            # train_loss += train_loss.item()

            iterations += 1
        
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_fmap_loss = sum(fmap_loss) / len(fmap_loss)
        avg_overlap_loss = sum(overlap_loss) / len(overlap_loss)
        avg_nce_loss = sum(nce_loss) / len(nce_loss)
            
        ### validation step  
        #TODO: early stopping
        val_loss = []
        val_fmap_loss = []
        val_overlap_loss = []
        val_nce_loss = []
        # val_loss = 0.0
        dpfm_net.eval()     # Optional when not using Model Specific layer
        for i, data in enumerate(valid_loader):
            data = shape_to_device(data, device)

            # NO data augmentation in validation step
            # data augmentation might consider set different params with the train_data
            # data = augment_batch(data, rot_x=30, rot_y=30, rot_z=60, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)
                
            # prepare iteration data
            C_gt = data["C_gt"].unsqueeze(0)
            map21 = data["map21"]
            gt_partiality_mask12, gt_partiality_mask21 = data["gt_partiality_mask12"], data["gt_partiality_mask21"]

            #calculate validation loss after each epoch
            C_pred, overlap_score12, overlap_score21, use_feat1, use_feat2 = dpfm_net(data)
            out, fmap, overlap, nce = criterion(C_gt, C_pred, map21, use_feat1, use_feat2,
                            overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21)

            val_loss.append(out.item())
            val_fmap_loss.append(fmap.item())
            val_overlap_loss.append(overlap.item())
            val_nce_loss.append(nce.item())
            # val_loss += val_loss.item() 

            # also add validation iterations to iterations
            iterations += 1

        avg_val_loss = sum(val_loss) / len(val_loss)
        avg_val_fmap_loss = sum(val_fmap_loss) / len(val_fmap_loss)
        avg_val_overlap_loss = sum(val_overlap_loss) / len(val_overlap_loss)
        avg_val_nce_loss = sum(val_nce_loss) / len(val_nce_loss)

        # print log every epoch instead of defined by iteration
        # if iterations % cfg["misc"]["log_interval"] == 0:
        # print(f"#epoch:{epoch}, #batch:{i + 1}, #iteration:{iterations}, train_loss:{avg_train_loss}, val_loss:{avg_val_loss}")
        print(f"#epoch:{epoch}, #iteration:{iterations}, train_loss:{avg_train_loss}, val_loss:{avg_val_loss}")

        # lr decay
        scheduler.step(avg_val_loss)    

        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        writer.add_scalar("FMap Loss/val", avg_val_fmap_loss, epoch)
        writer.add_scalar("FMap Loss/train", avg_fmap_loss, epoch)

        writer.add_scalar("Overlap Loss/val", avg_val_overlap_loss, epoch)
        writer.add_scalar("Overlap Loss/train", avg_overlap_loss, epoch)

        writer.add_scalar("NCE Loss/val", avg_val_nce_loss, epoch)
        writer.add_scalar("NCE Loss/train", avg_nce_loss, epoch)

        # save model
        if (epoch + 1) % cfg["misc"]["checkpoint_interval"] == 0:
            torch.save(dpfm_net.state_dict(), model_save_path.format(epoch))
    
    writer.flush()
    
def train_net_unsup(cfg, n_samples=None):
    """
    Rewritten train_net() function from DPFM to take more data classes
    """
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = cfg["dataset"]["cache_dir"]
    dataset_path = cfg["dataset"]["root_train"]

    save_dir_name = f'saved_models_{cfg["dataset"]["subset"]}_{cfg["dataset"]["model_name"]}'
    model_save_path = os.path.join(base_path, f"data/{save_dir_name}/ep" + "_{}.pth")
    if not os.path.exists(os.path.join(base_path, f"data/{save_dir_name}/")):
        os.makedirs(os.path.join(base_path, f"data/{save_dir_name}/"))

    # create dataset
    #TODO: implement training / validation split 
    if cfg["dataset"]["name"] == "shrec16":
        train_dataset = ShrecPartialDataset(dataset_path, name=cfg["dataset"]["subset"], k_eig=cfg["fmap"]["k_eig"],
                                            n_fmap=cfg["fmap"]["n_fmap"], use_cache=True, op_cache_dir=op_cache_dir)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=True)
    elif cfg["dataset"]["name"] == "tosca":
        train_dataset = Tosca(dataset_path, name=cfg["dataset"]["subset"], k_eig=cfg["fmap"]["k_eig"],
                                            n_fmap=cfg["fmap"]["n_fmap"], n_samples=n_samples, use_cache=True, op_cache_dir=op_cache_dir, use_adj=True)
        
        #TODO set train and validation ratio
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train, val = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        #Can increase num_workers to speed up training
        train_loader = torch.utils.data.DataLoader(train, batch_size=None, shuffle=True, num_workers=0)
        
        valid_loader = torch.utils.data.DataLoader(val, batch_size=None, shuffle=True, num_workers=0)
    elif cfg["dataset"]["name"] == "faust":
        raise NotImplementedError("FAUST support will come soon!")
    else:
        raise NotImplementedError("dataset not implemented!")

    # define model
    dpfm_net = DPFMNet_unsup(cfg).to(device)
    lr = float(cfg["optimizer"]["lr"])
    optimizer = torch.optim.Adam(dpfm_net.parameters(), lr=lr, betas=(cfg["optimizer"]["b1"], cfg["optimizer"]["b2"]))
    criterion = DPFMLoss_unsup(w_orth_C1=cfg["loss_unsup"]["w_orth_C1"], w_orth_C2=cfg["loss_unsup"]["w_orth_C2"],
    w_bij=cfg["loss_unsup"]["w_bij"], w_diff=cfg["loss_unsup"]["w_diff"]).to(device) 

    # Training loop
    print("start training")
    iterations = 0
    lrs = []
    lambda1 = lambda epoch: 0.65 ** epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, verbose=True)
    writer = SummaryWriter()

    for epoch in tqdm.tqdm(range(1, cfg["training"]["epochs"] + 1)):
        dpfm_net.train()
    
        ### training step
        # train_loss = 0.0
        train_loss = []
        orth_loss_C1 = []
        orth_loss_C2 = []
        bij_loss = []
        for i, data in enumerate(train_loader):
            data = shape_to_device(data, device)

            # data augmentation
            data = augment_batch(data, rot_x=30, rot_y=30, rot_z=60, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)

            # prepare iteration data

            # do iteration
            C1_pred, C2_pred, _, _ = dpfm_net(data)
            _, k1, k2 = C1_pred.shape
            eval1 = data["shape1"]["evals"][:k1].unsqueeze(0)
            eval2 = data["shape1"]["evals"][:k2].unsqueeze(0)
            out, orth_C1, orth_C2, bij = criterion(C1_pred, C2_pred, eval1, eval2)# , use_feat1, use_feat2
            
            out.backward()
            torch.nn.utils.clip_grad_norm_(dpfm_net.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(out.item())
            orth_loss_C1.append(orth_C1.item())
            orth_loss_C2.append(orth_C2.item())
            bij_loss.append(bij.item())
            # train_loss += train_loss.item()

            iterations += 1
        
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_orth_loss_C1 = sum(orth_loss_C1) / len(orth_loss_C1)
        avg_orth_loss_C2 = sum(orth_loss_C2) / len(orth_loss_C2)
        avg_bij_loss = sum(bij_loss) / len(bij_loss)
        
            
        ### validation step  
        #TODO: early stopping
        val_loss = []
        val_orth_loss_C1 = []
        val_orth_loss_C2 = []
        val_bij_loss = []
        # val_loss = 0.0
        dpfm_net.eval()     # Optional when not using Model Specific layer
        for i, data in enumerate(valid_loader):
            data = shape_to_device(data, device)

            # NO data augmentation in validation step
            # data augmentation might consider set different params with the train_data
            # data = augment_batch(data, rot_x=30, rot_y=30, rot_z=60, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)

            #calculate validation loss after each epoch
            C1_pred, C2_pred, _, _ = dpfm_net(data)
            _, k1, k2 = C1_pred.shape
            eval1 = data["shape1"]["evals"][:k1].unsqueeze(0)
            eval2 = data["shape1"]["evals"][:k2].unsqueeze(0)
            out, orth_C1, orth_C2, bij = criterion(C1_pred, C2_pred, eval1, eval2)# , use_feat1, use_feat2
            val_loss.append(out.item())
            val_orth_loss_C1.append(orth_C1.item())
            val_orth_loss_C2.append(orth_C2.item())
            val_bij_loss.append(bij.item())
            # val_loss += val_loss.item() 

            # also add validation iterations to iterations
            iterations += 1

        avg_val_loss = sum(val_loss) / len(val_loss)
        avg_val_orth_loss_C1 = sum(val_orth_loss_C1) / len(val_orth_loss_C1)
        avg_val_orth_loss_C2 = sum(val_orth_loss_C2) / len(val_orth_loss_C2)
        avg_val_bij_loss = sum(val_bij_loss) / len(val_bij_loss)

        # print log every epoch instead of defined by iteration
        # if iterations % cfg["misc"]["log_interval"] == 0:
        # print(f"#epoch:{epoch}, #batch:{i + 1}, #iteration:{iterations}, train_loss:{avg_train_loss}, val_loss:{avg_val_loss}")
        print(f"#epoch:{epoch}, #iteration:{iterations}, train_loss:{avg_train_loss}, val_loss:{avg_val_loss}")

        # lr decay
        scheduler.step(avg_val_loss)    

        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        writer.add_scalar("Orthogonality Loss C1/val", avg_val_orth_loss_C1, epoch)
        writer.add_scalar("Orthogonality Loss C1/train", avg_orth_loss_C1, epoch)
        
        writer.add_scalar("Orthogonality Loss C2/val", avg_val_orth_loss_C2, epoch)
        writer.add_scalar("Orthogonality Loss C2/train", avg_orth_loss_C2, epoch)

        writer.add_scalar("Bijectivity Loss/val", avg_val_bij_loss, epoch)
        writer.add_scalar("Bijectivity Loss/train", avg_bij_loss, epoch)


        # save model
        if (epoch + 1) % cfg["misc"]["checkpoint_interval"] == 0:
            torch.save(dpfm_net.state_dict(), model_save_path.format(epoch))
    
    writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of DPFM model.")

    parser.add_argument("--config", type=str, default="shrec16_cuts", help="Config file name")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"../dpfm/config/{args.config}.yaml", "r"))
    train_net(cfg)
