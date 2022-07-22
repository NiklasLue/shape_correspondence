import torch
from dann import train
from dann import test
from dann.model import DPFMNet_DA
from project.datasets import ShrecPartialDataset, Tosca, shape_to_device

def main_test(source_domain_path, target_domain_path, model_path, cfg, predictions_name, n_samples=None):
    
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

    source_test_loader = torch.utils.data.DataLoader(source_dataset, batch_size=None, shuffle=False, num_workers=0) 

    ########################
    ## Target dataloader  ##
    ########################
    
    target_dataset = Tosca(target_domain_path, name=cfg["dataset"]["subset"], k_eig=cfg["fmap"]["k_eig"],
                                        n_fmap=cfg["fmap"]["n_fmap"], n_samples=n_samples, use_cache=True,
                           op_cache_dir=op_cache_dir, use_adj=True)

    
    target_test_loader = torch.utils.data.DataLoader(target_dataset, batch_size=None, shuffle=True, num_workers=0)

    
   
    
    ########################
    ##  Training source   ##
    ######################## 
    
    pred_p2p_list_tgt, distances_tgt = test_target(cfg, target_test_loader, model_path, save_name, predictions_name)    
    pred_p2p_list_src, distances_src = test_source(cfg, source_test_loadet, model_path, save_name, predictions_name)
    torch.cuda.empty_cache()

    ########################
    ##    Training dann   ##
    ########################
    
    return pred_p2p_list_tgt, pred_p2p_list_src


if __name__ == "__main__":
    main()