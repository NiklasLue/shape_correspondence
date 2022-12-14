import os
import argparse
import yaml

import tqdm
import torch

from torch.utils.tensorboard import SummaryWriter


from dpcfm.dpcfm_model import DPCFMNet, DPCFMNetV2
from  dpcfm.dpcfm_utils import DPCFMLoss, DPCFMLossV2

from dpfm.utils import augment_batch

from project.datasets import ShrecPartialDataset, Tosca, shape_to_device


def train_net(cfg, v=1):
    """
    Rewritten train_net() function from DPCFM to take more data classes
    """
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = cfg["dataset"]["cache_dir"]
    dataset_path = cfg["dataset"]["root_train"]

    save_dir_name = f'saved_models_{cfg["dataset"]["subset"]}_3'
    model_save_path = os.path.join(base_path, f"data/{save_dir_name}/ep" + "_{}.pth")
    if not os.path.exists(os.path.join(base_path, f"data/{save_dir_name}/")):
        os.makedirs(os.path.join(base_path, f"data/{save_dir_name}/"))

    # create dataset
    if cfg["dataset"]["name"] == "shrec16":
        train_dataset = ShrecPartialDataset(dataset_path, name=cfg["dataset"]["subset"], k_eig=cfg["fmap"]["k_eig"],
                                            n_fmap=cfg["fmap"]["n_fmap"], use_cache=True, op_cache_dir=op_cache_dir)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=True)
    elif cfg["dataset"]["name"] == "tosca":
        train_dataset = Tosca(dataset_path, name=cfg["dataset"]["subset"], k_eig=cfg["fmap"]["k_eig"],
                                            n_fmap=cfg["fmap"]["n_fmap"], use_cache=True, op_cache_dir=op_cache_dir, use_adj=True)
        
        
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train, val = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        #Can increase num_workers to speed up training
        train_loader = torch.utils.data.DataLoader(train, batch_size=None, shuffle=True, num_workers=0)
        
        valid_loader = torch.utils.data.DataLoader(val, batch_size=None, shuffle=False, num_workers=0)
    elif cfg["dataset"]["name"] == "faust":
        raise NotImplementedError("FAUST support will come soon!")
    else:
        raise NotImplementedError("dataset not implemented!")

    # define model
    if v==1:
        dpcfm_net = DPCFMNet(cfg).to(device)
    elif v==2:
        dpcfm_net = DPCFMNetV2(cfg).to(device)
    else:
        raise ValueError(f"Selected version {v}: not available!")

    lr = float(cfg["optimizer"]["lr"])
    optimizer = torch.optim.Adam(dpcfm_net.parameters(), lr=lr, betas=(cfg["optimizer"]["b1"], cfg["optimizer"]["b2"]))
    torch.nn.utils.clip_grad_norm_(dpcfm_net.parameters(), 1)

    if v==1:
        criterion = DPCFMLoss(w_fmap=cfg["loss"]["w_fmap"], w_acc=cfg["loss"]["w_acc"], w_nce=cfg["loss"]["w_nce"],
                         nce_t=cfg["loss"]["nce_t"], nce_num_pairs=cfg["loss"]["nce_num_pairs"]).to(device)
    elif v==2:
        criterion = DPCFMLossV2(w_fmap=cfg["loss"]["w_fmap"], w_acc=cfg["loss"]["w_acc"], w_nce=cfg["loss"]["w_nce"],
                         nce_t=cfg["loss"]["nce_t"], nce_num_pairs=cfg["loss"]["nce_num_pairs"]).to(device)

    # Training loop
    print("start training")
    iterations = 0
    lrs = []
    lambda1 = lambda epoch: 0.65 ** epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, verbose=True)
    writer = SummaryWriter()

    for epoch in tqdm.tqdm(range(1, cfg["training"]["epochs"] + 1)):
        dpcfm_net.train()
    
        ### training step
        # train_loss = 0.0
        train_loss = []
        fmap_loss = []
        overlap_loss = []
        nce_loss = []
        coup_loss = []
        for i, data in enumerate(train_loader):
            data = shape_to_device(data, device)
            # data augmentation
            data = augment_batch(data, rot_x=30, rot_y=30, rot_z=60, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)

            # prepare iteration data
            C_gt = data["C_gt"].unsqueeze(0)
            C_gt2 = data["C2_gt"].unsqueeze(0)

            map21 = data["map21"]
            gt_partiality_mask12, gt_partiality_mask21 = data["gt_partiality_mask12"], data["gt_partiality_mask21"]

            # do iteration
            C_pred1, C_pred2, overlap_score12, overlap_score21, use_feat1, use_feat2 = dpcfm_net(data)
            
            _, k1, k2 = C_pred1.shape
            evals1 = data["shape1"]["evals"][:k1].unsqueeze(0)
            evals2 = data["shape1"]["evals"][:k2].unsqueeze(0)
            out, fmap, overlap, nce, coup = criterion(C_pred1, C_pred2, C_gt, C_gt2, map21, use_feat1, use_feat2, evals1, evals2,
                             overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21)
            
            out.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(out.item())
            fmap_loss.append(fmap.item())
            overlap_loss.append(overlap.item())
            nce_loss.append(nce.item())
            coup_loss.append(coup.item())
            # train_loss += train_loss.item()

            iterations += 1

        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_fmap_loss = sum(fmap_loss) / len(fmap_loss)
        avg_overlap_loss = sum(overlap_loss) / len(overlap_loss)
        avg_nce_loss = sum(nce_loss) / len(nce_loss)
        avg_coup_loss = sum(coup_loss) / len(coup_loss)
        
        
        
            
        ### validation step  
        #TODO: early stopping
        val_loss = []
        val_fmap_loss = []
        val_overlap_loss = []
        val_nce_loss = []
        val_coup_loss = []
        # val_loss = 0.0
        dpcfm_net.eval() 

        # Optional when not using Model Specific layer
        for i, data in enumerate(valid_loader):
            data = shape_to_device(data, device)

            # NO data augmentation in validation step
            # data augmentation might consider set different params with the train_data
            # data = augment_batch(data, rot_x=30, rot_y=30, rot_z=60, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)
                
            # prepare iteration data
            C_gt = data["C_gt"].unsqueeze(0)
            C_gt2 = data["C2_gt"].unsqueeze(0)
            map21 = data["map21"]
            gt_partiality_mask12, gt_partiality_mask21 = data["gt_partiality_mask12"], data["gt_partiality_mask21"]

            #calculate validation loss after each epoch
            C_pred1, C_pred2, overlap_score12, overlap_score21, use_feat1, use_feat2 = dpcfm_net(data)
            _, k1, k2 = C_pred1.shape
            evals1 = data["shape1"]["evals"][:k1].unsqueeze(0)
            evals2 = data["shape1"]["evals"][:k2].unsqueeze(0)
            out, fmap, overlap, nce, coup = criterion(C_pred1, C_pred2, C_gt, C_gt2, map21, use_feat1, use_feat2, evals1, evals2,
                            overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21)

            val_loss.append(out.item())
            val_fmap_loss.append(fmap.item())
            val_overlap_loss.append(overlap.item())
            val_nce_loss.append(nce.item())
            val_coup_loss.append(coup.item())
            # val_loss += val_loss.item() 

            # also add validation iterations to iterations
            iterations += 1




        avg_val_loss = sum(val_loss) / len(val_loss)
        avg_val_fmap_loss = sum(val_fmap_loss) / len(val_fmap_loss)
        avg_val_overlap_loss = sum(val_overlap_loss) / len(val_overlap_loss)
        avg_val_nce_loss = sum(val_nce_loss) / len(val_nce_loss)
        avg_val_coup_loss = sum(val_coup_loss) / len(val_coup_loss)

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
        
        writer.add_scalar("Coup Loss/val", avg_val_coup_loss, epoch)
        writer.add_scalar("Coup Loss/train", avg_coup_loss, epoch) 

        

        # save model
        if (epoch + 1) % cfg["misc"]["checkpoint_interval"] == 0:
            torch.save(dpcfm_net.state_dict(), model_save_path.format(epoch))
    
    writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of DPCFM model.")

    parser.add_argument("--config", type=str, default="shrec16_cuts", help="Config file name")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"../dpfm/config/{args.config}.yaml", "r"))
    train_net(cfg)
