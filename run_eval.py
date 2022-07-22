import yaml
import os
import argparse

from project.eval_dpfm import eval_net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='absolute path to files', default=os.path.dirname(os.path.realpath(__file__)), dest="path")
    parser.add_argument('-d', '--data_path', type=str, help='absolute path to data', default="../..", dest="data_path")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(f"{args.path}/dpfm/config/shrec16_cuts.yaml", "r"))
    cfg["dataset"]["root_train"] = cfg["dataset"]["root_train"].replace("{{ABS_PATH}}", args.data_path)
    cfg["dataset"]["root_test"] = cfg["dataset"]["root_test"].replace("{{ABS_PATH}}", args.data_path)
    cfg["dataset"]["cache_dir"] = cfg["dataset"]["cache_dir"].replace("{{ABS_PATH}}", args.data_path)
    eval_net(cfg, f"{args.path}/project/data/shrec_pretrained_model/.pth", f"{args.path}/data/own_pt_cfm.pt", mode="CFM")
