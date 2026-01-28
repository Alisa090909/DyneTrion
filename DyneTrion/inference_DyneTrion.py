
import os
import torch
import GPUtil
import time
import numpy as np
import hydra
import logging
import copy
import random
import pandas as pd
from icecream import ic
import sys
from datetime import datetime
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils import data
from typing import Dict

from src.data import DyneTrion_data_loader_dynamic
from src.data import utils as du
import DyneTrion.train_DyneTrion as train_DyneTrion


class Evaluator:
    def __init__(
            self,
            conf: DictConfig,
            conf_overrides:Dict=None
    ):
        self._log = logging.getLogger(__name__)

        # Remove static type checking.
        OmegaConf.set_struct(conf, False)

        # Prepare configs.
        self._conf = conf
        self._eval_conf = conf.eval
        self._diff_conf = conf.diffuser
        self._data_conf = conf.data
        self._exp_conf = conf.experiment

        # Set-up GPU
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        self._log.info(f'Using device: {self.device}')
        # model weight
        self._weights_path = self._eval_conf.weights_path
        output_dir =self._eval_conf.output_dir
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)
        self._log.info(f'Saving results to {self._output_dir}')
        # Load models and experiment
        self._load_ckpt(conf_overrides)


        

    def _load_ckpt(self, conf_overrides):
        """Loads in model checkpoint."""
        self._log.info(f'===================>>>>>>>>>>>>>>>> Loading weights from {self._weights_path}')
        # Read checkpoint and create experiment.
        weights_pkl = du.read_pkl(self._weights_path, use_torch=True, map_location='cpu')#self.device

        # Merge base experiment config with checkpoint config.
        # self._conf.model = OmegaConf.merge(self._conf.model, weights_pkl['conf'].model)
        if conf_overrides is not None:
            self._conf = OmegaConf.merge(self._conf, conf_overrides)

        # Prepare model
        self._conf.experiment.ckpt_dir = None
        self._conf.experiment.warm_start = None
        self.exp = train_DyneTrion.Experiment(conf=self._conf)
        self.model = self.exp.model

        # Remove module prefix if it exists.
        model_weights = weights_pkl['model']
        model_weights = {k.replace('module.', ''):v for k,v in model_weights.items()}

        self.model.load_state_dict(model_weights)

        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(">>> Compiling the main model... This may take a few minutes.")
        self.model = torch.compile(self.model) 
        
        self.diffuser = self.exp.diffuser

        self._log.info(f"Loading model Successfully from {self._weights_path}!!!")

    def create_dataset(self,is_random=False):
        test_dataset = DyneTrion_data_loader_dynamic.PdbDataset(
            data_conf=self._data_conf,
            diffuser=self.exp._diffuser,
            is_training=False,
            is_testing=True,
            is_random_test=is_random,
            rank=self._conf.eval.rank_idx,
            grouped=self._conf.eval.group,
        )
        return test_dataset

    def start_evaluation(self):
        # define data process
        # we need to call the MD simulation to get the data
        # maybe add some func in the dateset class
        
        
        test_dataset = self.create_dataset(is_random=self._conf.eval.random_sample)

        eval_dir = self._output_dir
        os.makedirs(eval_dir, exist_ok=True)
        ic(eval_dir)
        config_path = os.path.join(eval_dir ,'eval_conf.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._conf, f=f)
        self._log.info(f'Saving inference config to {config_path}')
        print(self._conf.experiment)
        print('='*10, 'Eval Extrapolation')
        # self.exp.eval_extension(eval_dir,test_loader,self.device,noise_scale=self._exp_conf.noise_scale)
        self.exp._set_seed(42)
        pdb_base_path, ref_base_path = self.exp._prepare_extension_eval_dirs(eval_dir)
        extrapolation_time = self.exp._conf.eval.extrapolation_time
        for i in range(len(test_dataset)):
            valid_feats, pdb_names = test_dataset._get_row(i)
            for k,v in valid_feats.items():
                valid_feats[k] = v.unsqueeze(0)
            self.exp._process_one_protein_extrapolation(
                extrapolation_time,
                valid_feats,
                [pdb_names],
                ref_base_path,
                pdb_base_path,
                device=self.device,
                noise_scale=self.exp._exp_conf.noise_scale,
            )

@hydra.main(version_base=None, config_path="./config", config_name="eval_DyneTrion")
def run(conf: DictConfig) -> None:
    
    torch.set_float32_matmul_precision('high')
    
    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    sampler = Evaluator(conf)

    sampler.start_evaluation()

    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
