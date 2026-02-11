
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
from contextlib import contextmanager
import torch.cuda.nvtx as nvtx
import concurrent.futures

from src.data import DyneTrion_data_loader_dynamic
from src.data import utils as du
import DyneTrion.train_DyneTrion0209 as train_DyneTrion

@contextmanager
def nvtx_range(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()

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
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._conf.experiment.device = self.device
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
        
        self.model = torch.compile(self.model, dynamic=True) 
        
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
    
    def warmup_model(self, warmup_feats, device):
        """专门处理预热的函数：准备假噪声并调用推理"""
        print("==== [Warmup] 正在初始化计算图和显存池... ====")
        
        # 为预热准备“假噪声”
        f_time, l_len = warmup_feats['res_mask'].shape
        z_rot_all = torch.randn(100, f_time, l_len, 3, device=self.device)
        z_trans_all = torch.randn(100, f_time, l_len, 3, device=self.device)
        # 调用干净的推理函数 (跑 2 步即可)
        with torch.no_grad():
            self.inference_fn(
                warmup_feats, 
                num_t=2, 
                min_t=0.01, 
                aux_traj=True, 
                # precomputed_noises=warmup_noise
                z_rot_all=z_rot_all,
                z_trans_all=z_trans_all
            )
            torch.cuda.synchronize()
            # torch.cuda.empty_cache() 
        print("==== [Warmup] 预热成功，显存已锁定 ====")

    def start_evaluation(self):
        # define data process
        # we need to call the MD simulation to get the data
        # maybe add some func in the dateset class
        
        print("开始准备数据和预热...")
        # with nvtx_range("Data_Loading_And_Parsing"):
        test_dataset = self.create_dataset(is_random=self._conf.eval.random_sample)
        
        num_to_run = len(test_dataset)
        print(f"本次计划推理蛋白数量: {num_to_run}")
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        # seq_len 最大的索引
        current_batch_df = test_dataset.csv.iloc[:num_to_run]
        max_idx_in_batch = current_batch_df['seq_len'].idxmax()
        max_len = current_batch_df['seq_len'].max()
        pdb_id_of_max = current_batch_df.loc[max_idx_in_batch, 'pdb_id']
        relative_idx = current_batch_df.index.get_loc(max_idx_in_batch)

        print(f"==== 预热：使用本次批次中最长的蛋白 [ID: {pdb_id_of_max}, 长度: {max_len}] ====")

        # 进行极限预热
        with torch.no_grad():
            warmup_feats, _ = test_dataset._get_row(relative_idx)
            for k, v in warmup_feats.items():
                if torch.is_tensor(v):
                    warmup_feats[k] = v.to(self.device)
                    
            f_time, l_len = warmup_feats['res_mask'].shape
            z_rot_all = torch.randn(100, f_time, l_len, 3, device=self.device)
            z_trans_all = torch.randn(100, f_time, l_len, 3, device=self.device)
            
            self.exp.inference_fn(warmup_feats,num_t=3,min_t=0.01,aux_traj=True,
                                  z_rot_all=z_rot_all,
                                  z_trans_all=z_trans_all
                                  )
            torch.cuda.synchronize()
            # torch.cuda.empty_cache() 
        print("==== [Warmup] 预热成功，显存已锁定 ====")    
        
        future = executor.submit(test_dataset._get_row, 0)    
        
        # print("开始精准性能采集...")
        # torch.cuda.profiler.start()

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
        
        
        for i in range(num_to_run):
            # valid_feats, pdb_names = test_dataset._get_row(i)
            valid_feats, pdb_names = future.result()
            if i + 1 < num_to_run:
                future = executor.submit(test_dataset._get_row, i + 1)
            for k,v in valid_feats.items():
                # valid_feats[k] = v.unsqueeze(0)
                valid_feats[k] = v.unsqueeze(0).to(self.device, non_blocking=True)
            # nvtx.range_push("_process_one_protein_extrapolation")
            self.exp._process_one_protein_extrapolation(
                extrapolation_time,
                valid_feats,
                [pdb_names],
                ref_base_path,
                pdb_base_path,
                device=self.device,
                noise_scale=self.exp._exp_conf.noise_scale,
                executor=executor,
            )
            # nvtx.range_pop()
        # torch.cuda.profiler.stop()
        executor.shutdown(wait=True)
        # print("性能采集结束。")

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
