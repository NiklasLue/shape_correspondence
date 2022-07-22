import torch
import numpy as np
import tqdm
import torch.optim as optim
import torch.nn as nn
import test
import os

from torch.utils.tensorboard import SummaryWriter

from dann.utils import save_model, set_model_mode, ExtLoss, FrobeniusLoss, DPFMLoss_da

from dann.model import FeatExtractorNet, MapExtractorNet, Discriminator, DPFMNet_DA
#from utils import visualize
from project.datasets import ShrecPartialDataset, Tosca, shape_to_device
from dpfm.utils import augment_batch



def source_only(cfg, source_train_loader, source_valid_loader, target_train_loader, target_valid_loader, save_name):
    #############################
    ##      Data-Loader        ##
    #############################
    
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = cfg["dataset"]["cache_dir"]
    #dataset_path = cfg["dataset"]["root_train"]

    save_dir_name = f'dann'
    model_save_path = os.path.join(base_path, f"data/{save_dir_name}/ep" + "_{}.pth")
    if not os.path.exists(os.path.join(base_path, f"data/{save_dir_name}/")):
        os.makedirs(os.path.join(base_path, f"data/{save_dir_name}/"))
    
    
    ################################
    #        Source training       #
    ################################

    print("Source-only training")
    iterations = 0

    dpfm_da_net = DPFMNet_DA(cfg).to(device)
    lr = float(cfg["optimizer"]["lr"])
    criterion = DPFMLoss_da().to(device)
    optimizer = optim.Adam(dpfm_da_net.parameters(), lr=lr, betas=(cfg["optimizer"]["b1"], cfg["optimizer"]["b2"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

    writer = SummaryWriter()
    for epoch in tqdm.tqdm(range(1, cfg["training"]["epochs"] + 1)):
        #print('Epoch : {}'.format(epoch))
        dpfm_da_net.train()

        start_steps = epoch * len(source_train_loader)
        total_steps = cfg["training"]["epochs"] * len(target_train_loader)
        
    
        ### training step
        train_loss = []
        fmap_loss = []
        overlap_loss = []
        nce_loss = []
        discriminator_loss = []
        for i, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):           
            source_data = shape_to_device(source_data, device)
            target_data = shape_to_device(target_data, device)
            
            p = float(i + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            # data augmentation
            source_data = augment_batch(source_data, rot_x=30, rot_y=30, rot_z=60, 
                                        std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)
            #target_data = augment_batch(target_data, rot_x=30, rot_y=30, rot_z=60, 
            #                            std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)
            size_src = [1,2]#len(source_data)

            label_src = torch.zeros(size_src).long().to(device)  

            # prepare iteration data
            C_gt = source_data["C_gt"].unsqueeze(0)
            map21 = source_data["map21"]
            gt_partiality_mask12, gt_partiality_mask21 = source_data["gt_partiality_mask12"], source_data["gt_partiality_mask21"]
                       

            ###############
            C_pred, src_domain_output, overlap_score12, overlap_score21, use_feat1, use_feat2  = dpfm_da_net(source_data, alpha=alpha)
            out, fmap, acc, nce, discriminator = criterion(C_gt, C_pred, map21, use_feat1, use_feat2, 
                                                              overlap_score12, overlap_score21, gt_partiality_mask12,
                                                              gt_partiality_mask21, src_domain_output, label_src)
            
            out.backward()
            torch.nn.utils.clip_grad_norm_(dpfm_da_net.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss.append(out.item())
            fmap_loss.append(fmap.item())
            overlap_loss.append(acc.item())
            nce_loss.append(nce.item())
            discriminator_loss.append(discriminator.item())
            iterations += 1

            # train_loss += train_loss.item()
        
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_fmap_loss = sum(fmap_loss) / len(fmap_loss)
        avg_overlap_loss = sum(overlap_loss) / len(overlap_loss)
        avg_nce_loss = sum(nce_loss) / len(nce_loss)
        avg_disc_loss = sum(discriminator_loss) / len(discriminator_loss)
        val_loss = []
        val_fmap_loss = []
        val_overlap_loss = []
        val_nce_loss = []
        val_discriminator_loss = []
        # val_loss = 0.0
        dpfm_da_net.eval()     # Optional when not using Model Specific layer
        with torch.no_grad():
            for i, (source_data, target_data) in enumerate(zip(source_valid_loader, target_valid_loader)):           
                source_data = shape_to_device(source_data, device)
                target_data = shape_to_device(target_data, device)
            
                p = float(i + start_steps) / total_steps
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
                # data augmentation
                #source_data = augment_batch(source_data, rot_x=30, rot_y=30, rot_z=60, 
                #                        std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)
                #target_data = augment_batch(target_data, rot_x=30, rot_y=30, rot_z=60, 
                #                            std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)
                size_src = [1,2]#len(source_data)

                label_src = torch.zeros(size_src).long().to(device)  
                # NO data augmentation in validation step
                # data augmentation might consider set different params with the train_data
                # data = augment_batch(data, rot_x=30, rot_y=30, rot_z=60, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)

                # prepare iteration data
                C_gt = source_data["C_gt"].unsqueeze(0)
                map21 = source_data["map21"]
                gt_partiality_mask12, gt_partiality_mask21 = source_data["gt_partiality_mask12"], source_data["gt_partiality_mask21"]


                ###############
                C_pred, src_domain_output, overlap_score12, overlap_score21, use_feat1, use_feat2  = dpfm_da_net(source_data,
                                                                                                                 alpha=alpha)
                out, fmap, acc, nce, discriminator = criterion(C_gt, C_pred, map21, use_feat1, use_feat2, 
                                                                  overlap_score12, overlap_score21, gt_partiality_mask12,
                                                                  gt_partiality_mask21, src_domain_output, label_src)
                val_loss.append(out.item())
                val_fmap_loss.append(fmap.item())
                val_overlap_loss.append(acc.item())
                val_nce_loss.append(nce.item())
                val_discriminator_loss.append(discriminator.item())

                # val_loss += val_loss.item() 

        avg_val_loss = sum(val_loss) / len(val_loss)
        avg_val_fmap_loss = sum(val_fmap_loss) / len(val_fmap_loss)
        avg_val_overlap_loss = sum(val_overlap_loss) / len(val_overlap_loss)
        avg_val_nce_loss = sum(val_nce_loss) / len(val_nce_loss)
        
        avg_val_discriminator_loss = sum(val_discriminator_loss) / len(val_discriminator_loss)
        
        avg_val_discr_loss = sum(val_discriminator_loss)/ len(val_discriminator_loss)
        print(f"#epoch:{epoch}, #iteration:{iterations}, train_loss:{avg_train_loss}, val_loss:{avg_val_loss}")
        scheduler.step(avg_train_loss)    

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        writer.add_scalar("FMap Loss/train", avg_fmap_loss, epoch)
        writer.add_scalar("FMap Loss/val", avg_val_fmap_loss, epoch)

        writer.add_scalar("Overlap Loss/train", avg_overlap_loss, epoch)
        writer.add_scalar("Overlap Loss/val", avg_val_overlap_loss, epoch)

        writer.add_scalar("NCE Loss/train", avg_nce_loss, epoch)
        writer.add_scalar("NCE Loss/val", avg_val_nce_loss, epoch)

        writer.add_scalar("Disc Loss/train", avg_disc_loss, epoch)
        writer.add_scalar("Disc Loss/val", avg_val_discr_loss, epoch)

    writer.flush()
    torch.cuda.empty_cache()

    save_model(dpfm_da_net, save_name)
    #visualize(encoder, 'source', save_name)


def dann(cfg, model, source_train_loader, source_valid_loader, target_train_loader, target_valid_loader, save_name):
    print("DANN training")
    
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    iterations = 0

    #map_ext_criterion = DPFMLoss().to(device)
    #discriminator_criterion = nn.CrossEntropyLoss().to(device)
    lr = float(cfg["optimizer"]["lr"])
    
    criterion = DPFMLoss_da().to(device)

    optimizer = optim.Adam(model.parameters(),
    lr=lr, betas=(cfg["optimizer"]["b1"], cfg["optimizer"]["b2"]))
    writer = SummaryWriter()
    torch.cuda.empty_cache()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

    
    for epoch in tqdm.tqdm(range(1, cfg["training"]["epochs"] + 1)):
        print('Epoch : {}'.format(epoch))
        model.train()
        start_steps = epoch * len(source_train_loader)
        total_steps = cfg["training"]["epochs"] * len(target_train_loader)
        torch.cuda.empty_cache()

        train_loss = []

        src_discriminator_loss = []
        tgt_discriminator_loss = []

        for i, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
            
            p = float(i + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            source_data = shape_to_device(source_data, device)
            target_data = shape_to_device(target_data, device)
            
            
            ##    This is wrong   ##
            
            size_src = [1,2]#len(source_data)
            size_tgt = [1,2]#len(target_data)
            
            #########################
            
            
            label_src = torch.zeros(size_src).long().to(device)  
            label_tgt = torch.ones(size_tgt).long().to(device)  
            
            # data augmentation
            source_data = augment_batch(source_data, rot_x=30, rot_y=30, rot_z=60, 
                                        std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)
            target_data = augment_batch(target_data, rot_x=30, rot_y=30, rot_z=60, 
                                        std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)
            

            #####################################


            optimizer.zero_grad()

            # prepare iteration data src
            C_gt = source_data["C_gt"].unsqueeze(0)
            map21 = source_data["map21"]
            gt_partiality_mask12, gt_partiality_mask21 = source_data["gt_partiality_mask12"], source_data["gt_partiality_mask21"]
            
            # train on source domain
            C_pred, src_domain_output, overlap_score12, overlap_score21, use_feat1, use_feat2  = model(source_data, alpha=alpha)
            out, fmap, acc, nce, src_discriminator = criterion(C_gt, C_pred, map21, use_feat1, use_feat2, 
                                                              overlap_score12, overlap_score21, gt_partiality_mask12,
                                                              gt_partiality_mask21, src_domain_output, label_src)
            

            # prepare iteration data tgt
            C_gt = target_data["C_gt"].unsqueeze(0)
            map21 = target_data["map21"]
            gt_partiality_mask12, gt_partiality_mask21 = target_data["gt_partiality_mask12"], target_data["gt_partiality_mask21"]
            
            # train on target domain
            C_pred, tgt_domain_output, overlap_score12, overlap_score21, use_feat1, use_feat2  = model(target_data, alpha=alpha)
            _, _, _, _, tgt_discriminator = criterion(C_gt, C_pred, map21, use_feat1, use_feat2, 
                                                              overlap_score12, overlap_score21, gt_partiality_mask12,
                                                              gt_partiality_mask21, tgt_domain_output, label_tgt)

            del overlap_score12, overlap_score21, use_feat1, use_feat2, C_pred
            loss = out + src_discriminator + tgt_discriminator
            iterations += 1
            
            # optimize dann
            loss.backward()
            optimizer.step()
            train_loss.append(out.item())
            
            src_discriminator_loss.append(src_discriminator.item())
            tgt_discriminator_loss.append(tgt_discriminator.item())
            
        avg_train_loss = sum(train_loss) / len(train_loss)
        
        avg_src_disc_loss = sum(src_discriminator_loss) / len(src_discriminator_loss)
        avg_tgt_disc_loss = sum(tgt_discriminator_loss) / len(tgt_discriminator_loss)

        val_loss_src = []
        val_loss_tgt = []
        val_fmap_loss = []
        val_overlap_loss = []
        val_nce_loss = []
        val_src_discriminator_loss = []     
        val_tgt_discriminator_loss = []     

        
        model.eval()     # Optional when not using Model Specific layer
        with torch.no_grad():
            for i, (source_data, target_data) in enumerate(zip(source_valid_loader, target_valid_loader)):           
                source_data = shape_to_device(source_data, device)
                target_data = shape_to_device(target_data, device)

                size_src = [1,2]#len(source_data)
                size_tgt = [1,2]#len(target_data)
                label_src = torch.zeros(size_src).long().to(device)  
                label_tgt = torch.ones(size_tgt).long().to(device)  
            
           
                p = float(i + start_steps) / total_steps
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
                # data augmentation
                #source_data = augment_batch(source_data, rot_x=30, rot_y=30, rot_z=60, 
                #                        std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)
                #target_data = augment_batch(target_data, rot_x=30, rot_y=30, rot_z=60, 
                #                            std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)
                size_src = [1,2]#len(source_data)

                label_src = torch.zeros(size_src).long().to(device)  
                # NO data augmentation in validation step
                # data augmentation might consider set different params with the train_data
                # data = augment_batch(data, rot_x=30, rot_y=30, rot_z=60, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)

                # prepare iteration data
                C_gt = source_data["C_gt"].unsqueeze(0)
                map21 = source_data["map21"]
                gt_partiality_mask12, gt_partiality_mask21 = source_data["gt_partiality_mask12"], source_data["gt_partiality_mask21"]


                ############### Source Validation
                C_pred_src, src_domain_output, overlap_score12, overlap_score21, use_feat1, use_feat2  = model(source_data,
                                                                                                                 alpha=alpha)
                out_src, fmap, acc, nce, src_discriminator = criterion(C_gt, C_pred_src, map21, use_feat1, use_feat2, 
                                                                  overlap_score12, overlap_score21, gt_partiality_mask12,
                                                                  gt_partiality_mask21, src_domain_output, label_src)
                
                ############### Target Validation
                C_pred_tgt, tgt_domain_output, overlap_score12, overlap_score21, use_feat1, use_feat2  = model(source_data,
                                                                                                                 alpha=alpha)
                out_tgt, fmap, acc, nce, tgt_discriminator = criterion(C_gt, C_pred_src, map21, use_feat1, use_feat2, 
                                                                  overlap_score12, overlap_score21, gt_partiality_mask12,
                                                                  gt_partiality_mask21, tgt_domain_output, label_src)
                val_loss_src.append(out_src.item())
                val_loss_tgt.append(out_tgt.item())

                val_fmap_loss.append(fmap.item())
                val_overlap_loss.append(acc.item())
                val_nce_loss.append(nce.item())
                val_src_discriminator_loss.append(src_discriminator.item())
                val_tgt_discriminator_loss.append(tgt_discriminator.item())        

        val_avg_src_train_loss = sum(val_loss_src) / len(val_loss_src)
        val_avg_tgt_train_loss = sum(val_loss_tgt) / len(val_loss_tgt)
        
        val_avg_src_disc_loss = sum(val_src_discriminator_loss) / len(val_src_discriminator_loss)
        val_avg_tgt_disc_loss = sum(val_tgt_discriminator_loss) / len(val_tgt_discriminator_loss)
        print(f"#epoch:{epoch}, train_loss:{avg_train_loss}, val_loss_src:{val_avg_src_train_loss}, val_loss_tgt:{val_avg_tgt_train_loss}")

        scheduler.step(avg_train_loss)    

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val",  val_avg_src_train_loss, epoch)
     

        writer.add_scalar("Disc src Loss/train", avg_src_disc_loss, epoch)
        writer.add_scalar("Disc src Loss/val", val_avg_src_disc_loss, epoch)
        
        writer.add_scalar("Disc tgt Loss/train", avg_tgt_disc_loss, epoch)
        writer.add_scalar("Disc tgt Loss/val", val_avg_tgt_disc_loss, epoch)
    writer.flush()        
          


    save_model(model, save_name)
    #visualize(encoder, 'source', save_name)

