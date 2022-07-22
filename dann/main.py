import torch
from dann import train
from dann.model import DPFMNet_DA
from project.datasets import ShrecPartialDataset, Tosca, shape_to_device


def main(source_domain_path, target_domain_path, cfg, n_samples=None):
    
    save_name = 'try_dann'
    
    
    op_cache_dir = cfg["dataset"]["cache_dir"]
    #dataset_path = cfg["dataset"]["root_train"]
    
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")
    
    
    ########################
    ## Source dataloader  ##
    ########################
    
    source_dataset = Tosca(source_domain_path, name=cfg["dataset"]["subset"], k_eig=cfg["fmap"]["k_eig"],
                                        n_fmap=cfg["fmap"]["n_fmap"], n_samples=n_samples, use_cache=True,
                           op_cache_dir=op_cache_dir, use_adj=True)

    #TODO set train and validation ratio
    train_size = int(0.8 * len(source_dataset))
    val_size = len(source_dataset) - train_size

    source_train, val = torch.utils.data.random_split(source_dataset, [train_size, val_size])

    #Can increase num_workers to speed up training
    source_train_loader = torch.utils.data.DataLoader(source_train, batch_size=None, shuffle=True, num_workers=0)

    source_valid_loader = torch.utils.data.DataLoader(val, batch_size=None, shuffle=False, num_workers=0)
    
    ########################
    ## Target dataloader  ##
    ########################
    
    target_dataset = Tosca(target_domain_path, name=cfg["dataset"]["subset"], k_eig=cfg["fmap"]["k_eig"],
                                        n_fmap=cfg["fmap"]["n_fmap"], n_samples=n_samples, use_cache=True,
                           op_cache_dir=op_cache_dir, use_adj=True)

    #TODO set train and validation ratio
    train_size = int(0.8 * len(target_dataset))
    val_size = len(target_dataset) - train_size

    target_train, val = torch.utils.data.random_split(target_dataset, [train_size, val_size])

    #Can increase num_workers to speed up training
    target_train_loader = torch.utils.data.DataLoader(target_train, batch_size=None, shuffle=True, num_workers=0)

    target_valid_loader = torch.utils.data.DataLoader(val, batch_size=None, shuffle=False, num_workers=0) 
    
   
    
    ########################
    ##  Training source   ##
    ######################## 
    
    #train.source_only(cfg, source_train_loader, source_valid_loader, target_train_loader, target_valid_loader, save_name)
    
    
    torch.cuda.empty_cache()

    ########################
    ##    Training dann   ##
    ########################
    
    model = DPFMNet_DA(cfg).to(device)
    model.load_state_dict(torch.load('trained_models/try1' + '_' + str(save_name) + '.pt'))
    
    train.dann(cfg, model, source_train_loader, source_valid_loader, target_train_loader, target_valid_loader, save_name)
    
    


if __name__ == "__main__":
    main()