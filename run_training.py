import yaml
import os
import argparse

from project.train_dpfm import train_net as train_dpfm
from project.train_dpfm import train_net_unsup
from project.train_dpcfm import train_net as train_dpcfm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='absolute path to files', default=os.path.dirname(os.path.realpath(__file__)), dest="path")
    parser.add_argument('-d', '--data_path', type=str, help='absolute path to data', default=os.path.dirname(os.path.realpath(__file__)), dest="data_path")
    parser.add_argument('-c', '--config', type=str, help='relative path to config file', default="dpfm/config/tosca_cuts.yaml", dest="config_path")
    parser.add_argument('-v', '--version', type=str, help='which model should be used in training', choices=['dpfm', 'unsupervised', 'dpcfm1', 'dpcfm2'], default='dpfm', dest='version')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(f"{args.path}/{args.config_path}", "r"))
    cfg["dataset"]["root_train"] = cfg["dataset"]["root_train"].replace("{{ABS_PATH}}", args.data_path)
    cfg["dataset"]["root_test"] = cfg["dataset"]["root_test"].replace("{{ABS_PATH}}", args.data_path)
    cfg["dataset"]["cache_dir"] = cfg["dataset"]["cache_dir"].replace("{{ABS_PATH}}", args.data_path)
    
    if args.version == 'dpfm':
        train_dpfm(cfg)
    elif args.version == 'unsupervised':
        train_net_unsup(cfg)
    elif args.version == 'dpcfm1':
        train_dpcfm(cfg, v=1)
    elif args.version == 'dpcfm2':
        train_dpcfm(cfg, v=2)
