import yaml
import os
import argparse

from project.eval_dpfm import eval_net, eval_net_unsup
from project.eval_dpcfm import eval_net as eval_net_dpcfm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='absolute path to files', default=os.path.dirname(os.path.realpath(__file__)), dest="path")
    parser.add_argument('-d', '--data_path', type=str, help='absolute path to data', default="../..", dest="data_path")
    parser.add_argument('-c', '--config', type=str, help='relative path to config file', default="project/config/tosca_cuts.yaml", dest="config_path")
    parser.add_argument('-mp', '--model_path', type=str, help='relative path to model file', default="project/models/dpfm.pth", dest="model_path")
    parser.add_argument('-m', '--mode', type=str, help='mode, one of CFM, FM', default="FM", dest="mode")
    parser.add_argument('-v', '--version', type=str, help='which model was used in training', choices=['dpfm', 'unsupervised', 'dpcfm1', 'dpcfm2'], default='dpfm', dest='version')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(f"{args.path}/{args.config_path}", "r"))
    cfg["dataset"]["root_train"] = cfg["dataset"]["root_train"].replace("{{ABS_PATH}}", args.data_path)
    cfg["dataset"]["root_test"] = cfg["dataset"]["root_test"].replace("{{ABS_PATH}}", args.data_path)
    cfg["dataset"]["cache_dir"] = cfg["dataset"]["cache_dir"].replace("{{ABS_PATH}}", args.data_path)

    if args.version == 'dpfm':
        eval_net(cfg, f"{args.path}/{args.model_path}", f"{args.path}/data/test.pt", mode=args.mode)
    elif args.version == 'unsupervised':
        eval_net_unsup(cfg, f"{args.path}/{args.model_path}", f"{args.path}/data/test.pt", mode=args.mode)
    elif args.version == 'dpcfm1':
        eval_net_dpcfm(cfg, f"{args.path}/{args.model_path}", f"{args.path}/data/test.pt", v=1)
    elif args.version == 'dpcfm2':
        eval_net_dpcfm(cfg, f"{args.path}/{args.model_path}", f"{args.path}/data/test.pt", v=2)
