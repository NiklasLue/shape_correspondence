import os
import argparse
import yaml

import tqdm
import torch


from dpfm.model import DPFMNet
from dpfm.utils import DPFMLoss, augment_batch
from project.datasets import ShrecPartialDataset, Tosca, shape_to_device

def train_net(cfg):
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

    save_dir_name = f'saved_models_{cfg["dataset"]["subset"]}'
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
                                            n_fmap=cfg["fmap"]["n_fmap"], use_cache=True, op_cache_dir=op_cache_dir, use_adj=True)
        
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
    dpfm_net = DPFMNet(cfg).to(device)
    lr = float(cfg["optimizer"]["lr"])
    optimizer = torch.optim.Adam(dpfm_net.parameters(), lr=lr, betas=(cfg["optimizer"]["b1"], cfg["optimizer"]["b2"]))
    criterion = DPFMLoss(w_fmap=cfg["loss"]["w_fmap"], w_acc=cfg["loss"]["w_acc"], w_nce=cfg["loss"]["w_nce"],
                         nce_t=cfg["loss"]["nce_t"], nce_num_pairs=cfg["loss"]["nce_num_pairs"]).to(device)

    # Training loop
    print("start training")
    iterations = 0
    lrs = []
    lambda1 = lambda epoch: 0.65 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    for epoch in tqdm.tqdm(range(1, cfg["training"]["epochs"] + 1)):


        dpfm_net.train()
    
        #training step
        train_loss = 0.0
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
            train_loss = criterion(C_gt, C_pred, map21, use_feat1, use_feat2,
                             overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21)
            
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += train_loss.item()
        
        # lr decay
        scheduler.step()    
            
        #validation step  
        
        #TODO: early stopping
        val_loss = 0.0
        dpfm_net.eval()     # Optional when not using Model Specific layer
        for i, data in enumerate(valid_loader):
            data = shape_to_device(data, device)

            # data augmentation might consider set different params with the train_data
            data = augment_batch(data, rot_x=30, rot_y=30, rot_z=60, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)
                
            # prepare iteration data
            C_gt = data["C_gt"].unsqueeze(0)
            map21 = data["map21"]
            gt_partiality_mask12, gt_partiality_mask21 = data["gt_partiality_mask12"], data["gt_partiality_mask21"]
            
            
            
            
            #calculate validation loss after each epoch
            C_pred, overlap_score12, overlap_score21, use_feat1, use_feat2 = dpfm_net(data)
            val_loss = criterion(C_gt, C_pred, map21, use_feat1, use_feat2,
                            overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21)

            
            val_loss += val_loss.item() 
            

            #log 
            iterations += 1
            if iterations % cfg["misc"]["log_interval"] == 0:
                print(f"#epoch:{epoch}, #batch:{i + 1}, #iteration:{iterations}, train_loss:{train_loss}, val_loss:{val_loss}")
        
       
        
           
        
        # save model
        if (epoch + 1) % cfg["misc"]["checkpoint_interval"] == 0:
            torch.save(dpfm_net.state_dict(), model_save_path.format(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of DPFM model.")

    parser.add_argument("--config", type=str, default="shrec16_cuts", help="Config file name")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"../dpfm/config/{args.config}.yaml", "r"))
    train_net(cfg)
