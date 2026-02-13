import sys
import os
import copy
import gc
import logging
import pickle
import random
import time
from collections import defaultdict
from datetime import datetime

import GPUtil

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tree
from Bio.SVDSuperimposer import SVDSuperimposer
from icecream import ic
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
import mdtraj as md
from contextlib import contextmanager
import torch.cuda.nvtx as nvtx

import swanlab
from tqdm import tqdm

from DyneTrion.utils import (
    compute_validation_metrics_all,
    plot_curve_merged,
    plot_rot_trans_curve,
    residue_constants,
)
from openfold.utils import rigid_utils as ru
from openfold.utils.loss import lddt_ca, torsion_angle_loss
from src.data import DyneTrion_data_loader_dynamic
from src.analysis import utils as au

from src.data import se3_diffuser as se3_diffuser
from src.data import all_atom as all_atom

from src.data import utils as du
from src.experiments import utils as eu
from src.model import diffusion_4d_network_dynamic
from src.toolbox.rot_trans_error import (
    average_quaternion_distances,
    average_translation_distances,
)

@contextmanager
def nvtx_range(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()

class Experiment:

    def __init__(
            self,
            *,
            conf: DictConfig,
        ):
        """Initialize experiment.

        Args:
            exp_cfg: Experiment configuration.
        """
        self._log = logging.getLogger(__name__)
        self._available_gpus = ''.join([str(x) for x in GPUtil.getAvailable(order='memory', limit = 8)])

        # Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            self._exp_conf.name = (f'{self._exp_conf.name}_{HydraConfig.get().job.num}')
        self._diff_conf = conf.diffuser
        self._model_conf = conf.model
        self._data_conf = conf.data
        self._use_recoder = self._exp_conf.use_recoder
        self._use_ddp = self._exp_conf.use_ddp
        self.dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        
        self.device = torch.device(self._conf.experiment.device)
        
        # 1. initialize ddp info if in ddp mode
        if self._use_ddp :
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            self.ddp_info = eu.get_ddp_info()
            if self.ddp_info['rank'] not in [0,-1]:
                self._log.addHandler(logging.NullHandler())
                self._log.setLevel("ERROR")
                self._use_recoder = False
                self._exp_conf.ckpt_dir = None
                
        self.trained_epochs = 0
        self.trained_steps = 0

        # Initialize experiment objects
        self._diffuser = se3_diffuser.SE3Diffuser(self._diff_conf)
        self._model = diffusion_4d_network_dynamic.FullScoreNetwork(self._model_conf, self.diffuser)

        num_parameters = sum(p.numel() for p in self._model.parameters())

        if self._conf.model.ipa.temporal and self._conf.model.ipa.frozen_spatial:
            self._log.info('Frozen model and only train temporal module')
            # only train motion module
            for param in self._model.parameters():
                param.requires_grad = False
            for name, param in self._model.named_parameters():
                if 'temporal' in name: # 'frame'
                    param.requires_grad = True

        trainable_num_parameters = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        self._exp_conf.num_parameters = num_parameters
        self._exp_conf.trainable_num_parameters  = num_parameters
        self._log.info(f'Number of model parameters {num_parameters}, trainable parameters:{trainable_num_parameters}')
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._exp_conf.learning_rate,amsgrad=True)
        if conf.experiment.warm_start:
            ckpt_path = conf.experiment.warm_start
            _, optimizer_his, epoch, step = self.load_pretrianed_model(ckpt_path=ckpt_path)

            if conf.experiment.reuse_step:
                print('Reuse Step')
                self.trained_epochs = epoch
                self.trained_steps = step
                print(f"loading model from: {ckpt_path}")
            # self._optimizer.load_state_dict(optimizer_his)
            
        self._init_log()
        self._init_best_eval()
        if not self.conf.experiment.training:
            seed = 0
        else:
            seed = dist.get_rank()
        self._set_seed(seed)
        
        self.num_t = self._data_conf.num_t 
        self.reverse_steps = torch.linspace(self._data_conf.min_t, 1.0, self.num_t, device=self.device).flip(0)
        self.t_placeholder = torch.ones((1,), device=self.device)
        self.dt = 1.0 / self.num_t
        self.sqrt_dt = torch.sqrt(torch.tensor(self.dt, device=self.device))
        
        self.diffuser._so3_diffuser.to_device(self.device)
        self.diffuser._r3_diffuser.to_device(self.device)
        
        # pre-compute Scaling
        with torch.no_grad():

            all_rot_s, all_trans_s = self.diffuser.score_scaling(self.reverse_steps)
            
            self.all_rot_scales = all_rot_s.to(self.device)   # [100]
            self.all_trans_scales = all_trans_s.to(self.device) # [100]
            
            self.sc_rot_scale = self.all_rot_scales[0]
            self.sc_trans_scale = self.all_trans_scales[0]
            self.sc_t = self.reverse_steps[0]
        
        
    def _init_best_eval(self):
        self.best_trained_steps = 0
        self.best_trained_epoch = 0
        self.best_rmse_ca = 10000
        self.best_rmse_all = 10000
        self.best_drmsd = 10000
        self.best_rmsd_ca_aligned = 10000
        self.best_rot_error=1000
        self.best_trans_error = 1000
        self.best_ref_rot_error = 1000
        self.best_ref_trans_error = 1000

    def _init_log(self):

        if self._exp_conf.ckpt_dir is not None:
            # Set-up checkpoint location
            ckpt_dir = os.path.join(
                self._exp_conf.ckpt_dir,
                # self._exp_conf.name,
                self.dt_string )
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            self._exp_conf.ckpt_dir = ckpt_dir
            self._log.info(f'Checkpoints saved to: {ckpt_dir}')
        else:  
            self._log.info('Checkpoint not being saved.')

        if self._exp_conf.eval_dir is not None :
            eval_dir = os.path.join(
                self._exp_conf.eval_dir,
                self._exp_conf.name,
                self.dt_string )
            self._exp_conf.eval_dir = eval_dir
            self._log.info(f'Evaluation saved to: {eval_dir}')
        else:
            self._exp_conf.eval_dir = os.devnull
            self._log.info(f'Evaluation will not be saved.')

    def load_pretrianed_model(self, ckpt_path):
        try:
            self._log.info(f'Loading checkpoint from {ckpt_path}')
            ckpt_pkl = torch.load(ckpt_path, map_location='cpu')

            if ckpt_pkl is not None and 'model' in ckpt_pkl:
                ckpt_model = ckpt_pkl['model']

                if ckpt_model is not None:
                    ckpt_model = {k.replace('module.', ''): v for k, v in ckpt_model.items()}
                    model_state_dict = self._model.state_dict()
                    # pretrained_dict = {k: v for k, v in ckpt_model.items() if k in model_state_dict}
                    pretrained_dict = {k: v for k, v in ckpt_model.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
                    model_state_dict.update(pretrained_dict)
                    self._model.load_state_dict(model_state_dict)
                    self._log.info(f'Warm starting from: {ckpt_path}')
                    # del ckpt_pkl,ckpt_model,pretrained_dict,model_state_dict
                    # gc.collect()
                    return ckpt_pkl['conf'], ckpt_pkl['optimizer'], ckpt_pkl['epoch'], ckpt_pkl['step']
                else:
                    self._log.error("Checkpoint model is None.")
                    return False
            else:
                self._log.error("Checkpoint or model not found in checkpoint file.")
                return False
        except Exception as e:
            self._log.error(f"Error loading checkpoint: {e}")
            return False


    @property
    def diffuser(self):
        return self._diffuser

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf

    def create_dataset(self):
        
        # Datasets
        train_dataset = DyneTrion_data_loader_dynamic.PdbDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=True
        )

        valid_dataset = DyneTrion_data_loader_dynamic.PdbDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=False
        )
        # Loaders
        num_workers = self._exp_conf.num_loader_workers

        persistent_workers = True if num_workers > 0 else False
        prefetch_factor=2
        prefetch_factor = 2 if num_workers == 0 else prefetch_factor

        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        train_loader = data.DataLoader(
                train_dataset,
                batch_size=self._exp_conf.batch_size if not self._exp_conf.use_ddp else self._exp_conf.batch_size // self.ddp_info['world_size'],
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                drop_last=False,
                sampler=sampler,
                multiprocessing_context='fork' if num_workers != 0 else None,
        )
        valid_loader = data.DataLoader(
                valid_dataset,
                batch_size=self._exp_conf.eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                drop_last=False,
                multiprocessing_context='fork' if num_workers != 0 else None,
        )



        return train_loader, valid_loader

    def init_swanlab_logger(self):
        self._log.info("Initializing SwanLab Recoder.")
        conf_dict = OmegaConf.to_container(self._conf, resolve=True)
        swanlab_mode = conf_dict["experiment"]["recoder"]["mode"]
        if swanlab_mode == "cloud":
            swanlab.login(api_key=conf_dict["experiment"]["recoder"]["api_key"])
        self.swanlab_logger = swanlab.init(
            project=conf_dict["experiment"]["project"],
            experiment_name=conf_dict["experiment"]["name"],
            config=conf_dict,
            mode=swanlab_mode,
            logdir=conf_dict["experiment"]["recoder"]["save_path"],
        )

    def start_training(self, return_logs=False):
        # Set environment variables for which GPUs to use.
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            replica_id = int(HydraConfig.get().job.num)
        else:
            replica_id = 0
        if self._use_recoder and replica_id == 0:
            self.init_swanlab_logger()
        assert(not self._exp_conf.use_ddp or self._exp_conf.use_gpu)
        # GPU mode
        if torch.cuda.is_available() and self._exp_conf.use_gpu:
            # single GPU mode
            if self._exp_conf.num_gpus==1 :
                gpu_id = self._available_gpus[replica_id]
                device = f"cuda:{gpu_id}"
                self._model = self.model.to(device)
                self._log.info(f"Using device: {device}")
            #muti gpu mode
            elif self._exp_conf.num_gpus > 1:
                device_ids = [f"cuda:{i}" for i in self._available_gpus[:self._exp_conf.num_gpus]]
                #DDP mode
                if self._use_ddp :
                    device = torch.device("cuda",self.ddp_info['local_rank'])
                    model = self.model.to(device)
                    self._model = DDP(model, device_ids=[self.ddp_info['local_rank']], output_device=self.ddp_info['local_rank'],find_unused_parameters=True)
                    self._log.info(f"Multi-GPU training on GPUs in DDP mode, node_id : {self.ddp_info['node_id']}, devices: {device_ids}")
                #DP mode
                else:
                    if len(self._available_gpus) < self._exp_conf.num_gpus:
                        raise ValueError(f"require {self._exp_conf.num_gpus} GPUs, but only {len(self._available_gpus)} GPUs available ")
                    self._log.info(f"Multi-GPU training on GPUs in DP mode: {device_ids}")
                    gpu_id = self._available_gpus[replica_id]
                    device = f"cuda:{gpu_id}"
                    self._model = DP(self._model, device_ids=device_ids)
                    self._model = self.model.to(device)
        else:
            device = 'cpu'
            self._model = self.model.to(device)
            self._log.info(f"Using device: {device}")

        # if self.conf.experiment.warm_start:
        #     for state in self._optimizer.state.values():
        #         for k, v in state.items():
        #             if torch.is_tensor(v):
        #                 state[k] = v.to(device)

        self._model.train()
        
        (train_loader,valid_loader) = self.create_dataset()

        logs = []
        # torch.cuda.empty_cache()
        for epoch in range(self.trained_epochs, self._exp_conf.num_epoch):
            self.trained_epochs = epoch
            train_loader.sampler.set_epoch(epoch)
            epoch_log = self.train_epoch(
                train_loader,
                valid_loader,
                device,
                return_logs=return_logs
            )
            # self._schedule.step()

            if return_logs:
                logs.append(epoch_log)
        if self._exp_conf.ckpt_dir is not None:
            ckpt_path = os.path.join(self._exp_conf.ckpt_dir, f'last_step_{self.trained_steps}.pth')
            du.write_checkpoint(
                ckpt_path,
                copy.deepcopy(self.model.state_dict()),
                self._conf,
                copy.deepcopy(self._optimizer.state_dict()),
                self.trained_epochs,
                self.trained_steps,
                logger=self._log,
                use_torch=True
            )
        self._log.info('Done')
        return logs

    def update_fn(self, data):
        """Updates the state using some data and returns metrics."""
        self._optimizer.zero_grad()
        loss, aux_data = self.loss_fn(data)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self._optimizer.step()

        return loss, aux_data

    def train_epoch(self, train_loader, valid_loader, device,return_logs=False):
        log_lossses = defaultdict(list)
        global_logs = []
        log_time = time.time()
        step_time = time.time()
        # Run evaluation
        
        for train_feats in train_loader:
            self.model.train()
            train_feats = tree.map_structure(lambda x: x.to(device), train_feats)
            
            # TODO flatten the dim of batch and frame_time sss
            for k in train_feats.keys():
                v = train_feats[k]
                if len(v.shape)>1:
                    reshaped_tensor = torch.flatten(v, start_dim=0, end_dim=1)#torch.reshape(v, (tensor.size(0), -1))
                    train_feats[k] = reshaped_tensor

            
            loss, aux_data = self.update_fn(train_feats)

            if return_logs:
                global_logs.append(loss)
            for k,v in aux_data.items():
                log_lossses[k].append(v)
            self.trained_steps += 1

            if self.trained_steps == 1 or self.trained_steps % self._exp_conf.log_freq == 0:
                elapsed_time = time.time() - log_time
                log_time = time.time()
                step_per_sec = self._exp_conf.log_freq / elapsed_time
                rolling_losses = tree.map_structure(np.mean, log_lossses)
                loss_log = ' '.join([
                    f'{k}={v[0]:.4f}'
                    for k,v in rolling_losses.items() if 'batch' not in k
                ])
                self._log.info(f'Epoch[{self.trained_epochs}/{self._exp_conf.num_epoch}] trained_steps: [{self.trained_steps}]: {loss_log}, steps/sec={step_per_sec:.5f}')
                log_lossses = defaultdict(list)

            # Take checkpoint
            if self._exp_conf.ckpt_dir is not None and ((self.trained_steps % self._exp_conf.ckpt_freq) == 0 or (self._exp_conf.early_ckpt and self.trained_steps == 100)):
                ckpt_path = os.path.join(self._exp_conf.ckpt_dir, f'step_{self.trained_steps}.pth')
                du.write_checkpoint(
                    ckpt_path,
                    copy.deepcopy(self.model.state_dict()),
                    self._conf,
                    copy.deepcopy(self._optimizer.state_dict()),
                    self.trained_epochs,
                    self.trained_steps,
                    logger=self._log,
                    use_torch=True
                )

                if self._exp_conf.enable_validation:
                    # Run evaluation
                    self._log.info(f'Running evaluation of {ckpt_path}')
                    start_time = time.time()
                    eval_dir = os.path.join(self._exp_conf.eval_dir, f'step_{self.trained_steps}')
                    os.makedirs(eval_dir, exist_ok=True)
                    results = self.eval_fn(eval_dir, valid_loader, device,
                        noise_scale=self._exp_conf.noise_scale
                    )
                    eval_time = time.time() - start_time
                    eval_logs = {"Eval/Eval_time": float(eval_time)}
                    
                    mean_metrics = results["metrics"].mean(numeric_only=True)
                    for metric_name, value in mean_metrics.items():
                        eval_logs[f"Eval/{metric_name}"] = float(value)
                    # Scalar metrics
                    eval_logs = {
                        "Eval/RigidError_rot_pred": float(results["rot_trans_error_mean"]["ave_rot"]),
                        "Eval/RigidError_rot_ref":  float(results["rot_trans_error_mean"]["first_rot"]),
                        "Eval/RigidError_trans_pred": float(results["rot_trans_error_mean"]["ave_trans"]),
                        "Eval/RigidError_trans_ref":  float(results["rot_trans_error_mean"]["first_trans"]),
                    }

                    # matplotlib Figure
                    eval_logs.update({
                        "Eval/Fig_dis_curve": swanlab.Image(results["fig_dis_curve"]),
                        "Eval/Fig_dis_curve_aligned": swanlab.Image(results["fig_dis_curve_aligned"]),
                        "Eval/Fig_error": swanlab.Image(results["fig_error"]),
                    })
                    swanlab.log(eval_logs, step=self.trained_steps)
                    
                    info_dict = {
                        "info/best_trained_steps": self.best_trained_steps,
                        "info/best_trained_epoch": self.best_trained_epoch,
                        "info/best_rmse_all": float(self.best_rmse_all),
                        "info/relat_rmse_ca": float(self.best_rmse_ca),
                        "info/rot_error": float(self.best_rot_error),
                        "info/ref_rot_error": float(self.best_ref_rot_error),
                        "info/trans_error": float(self.best_trans_error),
                        "info/ref_trans_error": float(self.best_ref_trans_error),
                        "info/relat_rmsd_ca_aligned": float(self.best_rmsd_ca_aligned),
                        "info/relat_drmsd": float(self.best_drmsd),
                    }
                    swanlab.log(info_dict, print_to_console=True)
                    self._log.info(f'Finished evaluation in {eval_time:.2f}s')

            # Remote log to tensorborad.
            if self._use_recoder:
                step_time = time.time() - step_time
                example_per_sec = self._exp_conf.batch_size / step_time
                step_time = time.time()
                # Logging basic metrics
                log_metrics = {
                    # Losses
                    "Train/Loss_total": loss,
                    "Train/Loss_rot": aux_data["rot_loss"],
                    "Train/Loss_trans": aux_data["trans_loss"],
                    "Train/Loss_torsion": aux_data["torsion_loss"],
                    "Train/Loss_bb_atom": aux_data["bb_atom_loss"],
                    "Train/Loss_dist_mat": aux_data["dist_mat_loss"],
                    # Rigid updates
                    "Train/Rigid_rot0": aux_data["update_rots"][0],
                    "Train/Rigid_rot1": aux_data["update_rots"][1], 
                    "Train/Rigid_rot2": aux_data["update_rots"][2],
                    "Train/Rigid_trans0": aux_data["update_trans"][0],
                    "Train/Rigid_trans1": aux_data["update_trans"][1],
                    "Train/Rigid_trans2": aux_data["update_trans"][2],
                    # Speed
                    "Train/Speed_examples_per_sec": float(example_per_sec),
                }
                
                
                bb_grads = [p.grad for name, p in self.model.named_parameters() if p.grad is not None and "bb_update" in name]
                if bb_grads:
                    bb_grads_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2.0) for g in bb_grads]), 2.0).item()
                    log_metrics["Train/Grad_bb_update"] = bb_grads_norm

                # Logging checkpoint metrics if available
                if torch.isnan(loss):
                    swanlab.log({"Alerts": f"Encountered NaN loss after {self.trained_epochs} epochs, {self.trained_steps} steps"},
                                step=self.trained_steps)
                    raise Exception("NaN encountered")
                
                swanlab.log(log_metrics, step=self.trained_steps)

        if return_logs:
            return global_logs

    def eval_fn(self, eval_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0,is_training=True):
        # === 初始化路径与指标 ===
        dirs = self._prepare_eval_dirs(eval_dir, is_training)
        metrics = self._init_eval_metrics()

        # === Evaluate each sample ===
        for valid_feats, pdb_names, start_index in tqdm(valid_loader, desc="Evaluating", ncols=100):
            result = self._process_one_protein_for_eval(
                valid_feats, pdb_names, start_index,
                device=device,
                dirs=dirs,
                min_t=min_t,
                num_t=num_t,
                noise_scale=noise_scale,
                is_training=is_training
            )
            self._accumulate_metrics(metrics, result)

        # === Generate summary results ===
        return self._finalize_eval_outputs(metrics, dirs, eval_dir)

    
    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
    def eval_extension(self, eval_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0,seed=42):
        self._set_seed(seed)
        # ergodic the validation
        pdb_base_path, ref_base_path = self._prepare_extension_eval_dirs(eval_dir)
        print(f"\n{'*' * 10} Protein number: {len(valid_loader)} {'*' * 10}")
        extrapolation_time = self._conf.eval.extrapolation_time
        
        for valid_feats, pdb_names, start_index  in valid_loader:
            self._process_one_protein_extrapolation(
                extrapolation_time,
                valid_feats,
                pdb_names,
                ref_base_path,
                pdb_base_path,
                device,
                min_t,
                num_t,
                noise_scale,
            )


    def _self_conditioning(self, batch,drop_ref=False):
        model_sc = self.model(batch,drop_ref=drop_ref,is_training = self._exp_conf.training)
        batch['sc_ca_t'] = model_sc['rigids'][..., 4:]
        return batch

    def loss_fn(self, batch):
        """Computes loss and auxiliary data.

        Args:
            batch: Batched data.
            model_out: Output of model ran on batch.

        Returns:
            loss: Final training loss scalar.
            aux_data: Additional logging data.
        """
        
        if self._model_conf.embed.embed_self_conditioning and random.random() > 0.5:
            with torch.no_grad():
                batch = self._self_conditioning(batch)
        model_out = self.model(batch, is_training=self._exp_conf.training)

        bb_mask = batch['res_mask']
        diffuse_mask = 1 - batch['fixed_mask']

        loss_mask = bb_mask * diffuse_mask
        batch_size, num_res = bb_mask.shape

        torsion_loss = torsion_angle_loss(
            a=model_out['angles'],
            a_gt=batch['torsion_angles_sin_cos'],
            a_alt_gt=batch['alt_torsion_angles_sin_cos'],
            mask=batch['torsion_angles_mask']) * self._exp_conf.torsion_loss_weight

        gt_rot_score = batch['rot_score']
        rot_score_scaling = batch['rot_score_scaling']

        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_score = model_out['rot_score'] * diffuse_mask[..., None]
        pred_trans_score = model_out['trans_score'] * diffuse_mask[..., None]

        # Translation x0 loss
        gt_trans_x0 = batch['rigids_0'][..., 4:] #* self._exp_conf.coordinate_scaling
        pred_trans_x0 = model_out['rigids'][..., 4:] #* self._exp_conf.coordinate_scaling

        ref_trans_loss = torch.sum(
            (gt_trans_x0 -  batch['ref_rigids_0'][..., 4:][-1].unsqueeze(0).expand_as(gt_trans_x0))**2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)
        ref_trans_loss *= self._exp_conf.trans_loss_weight

        trans_loss = torch.sum(
            (gt_trans_x0 - pred_trans_x0).abs() * loss_mask[..., None],
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)
        trans_loss *= self._exp_conf.trans_loss_weight
        rot_mse = (gt_rot_score - pred_rot_score)**2 * loss_mask[..., None]
        rot_loss = torch.sum(
            rot_mse / rot_score_scaling[:, None, None]**2,
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)
        rot_loss *= self._exp_conf.rot_loss_weight

        ref_rot_mse = (gt_rot_score - batch['ref_rot_score'][-1].unsqueeze(0).expand_as(gt_rot_score))**2 * loss_mask[..., None]
        ref_rot_loss = torch.sum(
            ref_rot_mse / rot_score_scaling[:, None, None]**2,
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)
        ref_rot_loss *= self._exp_conf.rot_loss_weight
        rot_loss *= batch['t'] > self._exp_conf.rot_loss_t_threshold

        rot_loss *= int(self._diff_conf.diffuse_rot)

        
        # Backbone atom loss
        pred_atom37 = model_out['atom37'][:, :, :5]
        gt_rigids = ru.Rigid.from_tensor_7(batch['rigids_0'].type(torch.float32))
        gt_psi = batch['torsion_angles_sin_cos'][..., 2, :]  # psi
        gt_atom37, atom37_mask, _, _ = all_atom.compute_backbone(gt_rigids, gt_psi) # psi

        gt_atom37 = gt_atom37
        atom37_mask = atom37_mask

        gt_atom37 = gt_atom37[:, :, :5]
        atom37_mask = atom37_mask[:, :, :5]

        gt_atom37 = gt_atom37.to(pred_atom37.device)
        atom37_mask = atom37_mask.to(pred_atom37.device)
        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
        bb_atom_loss = torch.sum(
            (pred_atom37 - gt_atom37)**2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3)
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
        bb_atom_loss *= self._exp_conf.bb_atom_loss_weight
        # TODO here delete the filter
        bb_atom_loss *= batch['t'] < self._exp_conf.bb_atom_loss_t_filter  
        

        bb_atom_loss *= self._exp_conf.aux_loss_weight


        gt_flat_atoms = gt_atom37.reshape([batch_size, num_res*5, 3])
        gt_pair_dists = torch.linalg.norm(gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_atom37.reshape([batch_size, num_res*5, 3])
        pred_pair_dists = torch.linalg.norm(pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 5))
        flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res*5])
        flat_res_mask = torch.tile(bb_mask[:, :, None], (1, 1, 5))
        flat_res_mask = flat_res_mask.reshape([batch_size, num_res*5])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        # No loss on anything >6A
        proximity_mask = gt_pair_dists < 6
        pair_dist_mask  = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum((gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)
        dist_mat_loss *= self._exp_conf.dist_mat_loss_weight
        dist_mat_loss *= batch['t'] < self._exp_conf.dist_mat_loss_t_filter
        dist_mat_loss *= self._exp_conf.aux_loss_weight

        batch_loss_mask = batch_loss_mask
        final_loss = (
            rot_loss
            + trans_loss
            + bb_atom_loss
            + dist_mat_loss
            + torsion_loss
        )
        def normalize_loss(x):
            return x.sum() /  (batch_loss_mask.sum() + 1e-10)
        aux_data = {
            'batch_train_loss': final_loss.detach(),
            'batch_rot_loss': rot_loss.detach(),
            'batch_trans_loss': trans_loss.detach(),
            'batch_bb_atom_loss': bb_atom_loss.detach(),
            'batch_dist_mat_loss': dist_mat_loss.detach(),
            'batch_torsion_loss':torsion_loss.detach(),
            'total_loss': normalize_loss(final_loss).detach(),
            'rot_loss': normalize_loss(rot_loss).detach(),
            'ref_rot_loss':normalize_loss(ref_rot_loss).detach(),
            'trans_loss': normalize_loss(trans_loss).detach(),
            'ref_trans_loss':normalize_loss(ref_trans_loss).detach(),
            'bb_atom_loss': normalize_loss(bb_atom_loss).detach(),
            'dist_mat_loss': normalize_loss(dist_mat_loss).detach(),
            'torsion_loss':normalize_loss(torsion_loss).detach(),
            'update_rots':torch.mean(torch.abs(model_out['rigid_update'][...,:3]),dim=(0,1)).detach(),
            'update_trans':torch.mean(torch.abs(model_out['rigid_update'][...,-3:]),dim=(0,1)).detach(),
        }

        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)

        return normalize_loss(final_loss), aux_data

    def _calc_trans_0(self, trans_score, trans_t, t):
        beta_t = self._diffuser._se3_diffuser._r3_diffuser.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (trans_score * cond_var + trans_t) / torch.exp(-1/2*beta_t)

    def _set_t_feats(self, feats, t, t_placeholder):
        feats['t'] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.diffuser.score_scaling(t)
        feats['rot_score_scaling'] = rot_score_scaling * t_placeholder
        feats['trans_score_scaling'] = trans_score_scaling * t_placeholder
        return feats

    def forward_traj(self, x_0, min_t, num_t):
        forward_steps = np.linspace(min_t, 1.0, num_t)[:-1]
        x_traj = [x_0]
        for t in forward_steps:
            x_t = self.diffuser.se3_diffuser._r3_diffuser.forward(
                x_traj[-1], t, num_t)
            x_traj.append(x_t)
        x_traj = torch.stack(x_traj, axis=0)
        return x_traj

    def inference_fn(
            self,
            data_init,
            num_t=None,
            min_t=None,
            center=True,
            aux_traj=False,
            self_condition=True,
            noise_scale=1.0,
            z_rot_all=None,
            z_trans_all=None,
        ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """
        self._model.eval()
        # Run reverse process.
        sample_feats = {k: v for k, v in data_init.items()} 
        
        # memory freezing
        rigids_buffer = sample_feats['rigids_t'].clone().contiguous()
        sample_feats['rigids_t'] = rigids_buffer 

        device = self.device
        reverse_steps = self.reverse_steps 
        t_placeholder = self.t_placeholder
        all_rot_scales = self.all_rot_scales
        all_trans_scales = self.all_trans_scales
        
        num_t = num_t if num_t is not None else self.num_t
        min_t = min_t if min_t is not None else self._data_conf.min_t
        
        # init rigids_t
        current_rigid_obj = ru.Rigid.from_tensor_7_fast(sample_feats['rigids_t'])
        
        all_rigids = []
        all_bb_prots = []
        t_start = time.perf_counter()
        with torch.no_grad():
            # self-condition
            if self._model_conf.embed.embed_self_conditioning and self_condition:
                sample_feats['t'] = self.sc_t * t_placeholder
                sample_feats['rot_score_scaling'] =self.sc_rot_scale * t_placeholder
                sample_feats['trans_score_scaling'] = self.sc_trans_scale * t_placeholder
                sample_feats = self._self_conditioning(sample_feats)
            for step_idx, t in enumerate(reverse_steps):
                # memory freezing
                rigids_buffer.copy_(current_rigid_obj.to_tensor_7().detach())
                # model infer
                if step_idx < len(reverse_steps) - 1: 
                    sample_feats['t'] = t * t_placeholder
                    sample_feats['rot_score_scaling'] = all_rot_scales[step_idx] * t_placeholder
                    sample_feats['trans_score_scaling'] = all_trans_scales[step_idx] * t_placeholder
                    model_out = self.model(sample_feats, is_training = self._exp_conf.training)
                    rot_score = model_out['rot_score']
                    trans_score = model_out['trans_score']
                    rigid_pred = model_out['rigids']
                    # use CFG inference
                    # if self._conf.model.cfg_drop_rate > 0.01:
                    #     model_out_uncond = self.model(sample_feats,drop_ref = True,is_training = self._exp_conf.training)
                    #     trans_score_unref = model_out_uncond["trans_score"]
                    #     rot_score_unref = model_out_uncond["rot_score"]
                    #     cfg_gamma = self._conf.model.cfg_gamma
                    #     trans_score = trans_score_unref + cfg_gamma*(trans_score-trans_score_unref)
                    if self._model_conf.embed.embed_self_conditioning:
                        sample_feats['sc_ca_t'] = rigid_pred[..., 4:]
                    fixed_mask = sample_feats['fixed_mask'] * sample_feats['res_mask']
                    diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
                    with autocast(dtype=torch.bfloat16):
                        current_rigid_obj = self.diffuser.reverse(
                            rigid_t=current_rigid_obj, 
                            rot_score=rot_score, 
                            trans_score=trans_score,
                            diffuse_mask=sample_feats['res_mask'],
                            t=t, 
                            dt=self.dt,
                            sqrt_dt=self.sqrt_dt,
                            z_rot=z_rot_all[step_idx],   
                            z_trans=z_trans_all[step_idx], 
                            center=center,
                            noise_scale=noise_scale,
                            device=self.device
                        )                 
                else:
                    model_out = self.model(sample_feats,is_training = self._exp_conf.training)
                    current_rigid_obj = ru.Rigid.from_tensor_7_fast(model_out['rigids'])
    
                # post process
                if aux_traj:
                    all_rigids.append(model_out['rigids'])
                
                if step_idx == len(reverse_steps)-1:
                    angles = model_out['angles']
                    gt_trans_0 = sample_feats['rigids_t'][..., 4:]
                    pred_trans_0 = rigid_pred[..., 4:]
                    trans_pred_0 = diffuse_mask[..., None] * pred_trans_0 + fixed_mask[..., None] * gt_trans_0
                    atom37_t = all_atom.compute_backbone_atom37(
                        bb_rigids=current_rigid_obj, 
                        aatypes=sample_feats['aatype'],
                        torsions = angles
                        )[0]
                    all_bb_prots.append(atom37_t)
                
        inference_time = time.perf_counter() - t_start
        print(f"inference_time:{inference_time:.2f} | num_t:{num_t} | noise_scale:{noise_scale}")
        
        def safe_flip(x):
            if len(x) > 0:
                return torch.flip(torch.stack(x), dims=(0,))
            return x 

        all_bb_prots = safe_flip(all_bb_prots)
        all_rigids = safe_flip(all_rigids)        
        
        ret = {
            'prot_traj': all_bb_prots, 
            'rigid_traj': all_rigids
        }
        return ret
    
    
    def _calc_rot_trans_error(self,pred_rigids,gt_rigids,ref_rigids):
        first_gt_rigids = ref_rigids
        pred_rigids = pred_rigids# move out the ref
        first_gt_rigids_expands = np.repeat(first_gt_rigids[np.newaxis, :, :], len(gt_rigids), axis=0)
        # pred out
        average_quat_distances = average_quaternion_distances(gt_rigids[...,:4], pred_rigids[...,:4])
        average_trans_distances = average_translation_distances(gt_rigids[...,4:], pred_rigids[...,4:],measurement='MAE')
        # ref frame out
        ref_average_quat_distances = average_quaternion_distances(gt_rigids[...,:4], first_gt_rigids_expands[...,:4])
        ref_average_trans_distances = average_translation_distances(gt_rigids[...,4:], first_gt_rigids_expands[...,4:],measurement='MAE')
        # caculate relative motion
        time_rot_dif = average_quaternion_distances(gt_rigids[...,:4], np.roll(gt_rigids[...,:4],shift=1,axis=0))
        time_trans_dif = average_translation_distances(gt_rigids[...,4:], np.roll(gt_rigids[...,4:],shift=1,axis=0),measurement='MAE')

        return average_quat_distances,average_trans_distances,ref_average_quat_distances,ref_average_trans_distances,time_rot_dif,time_trans_dif

    def _prepare_eval_dirs(self, eval_dir, is_training):
        """Create and return all necessary evaluation directories."""
        dirs = {
            "sample": os.path.join(eval_dir, "sample"),
            "gt": os.path.join(eval_dir, "gt"),
            "pred_npz": os.path.join(eval_dir, "pred_npz") if not is_training else None,
        }
        for path in dirs.values():
            if path:
                os.makedirs(path, exist_ok=True)
        return dirs
    
    def _init_eval_metrics(self):
        """Initialize containers for evaluation metrics."""
        return {
            "metric_list": [],
            "metric_all_list": [],
            "metric_aligned_list": [],
            "metric_aligned_all_list": [],
            "first_frame_all_list": [],
            "save_name_list": [],
            "start_index_list": [],
            "rot_trans_error_dict": {
                "name": [], "ave_rot": [], "ave_trans": [],
                "first_rot": [], "first_trans": [],
                "time_rot_dif": [], "time_trans_dif": [],
            },
            "save_pdb_dict": {},
        }

    def _process_one_protein_for_eval(
        self,
        valid_feats,
        pdb_names,
        start_index,
        device,
        dirs,
        min_t=None,
        num_t=None,
        noise_scale=1.0,
        is_training=True,
    ):
        """Evaluate a single protein and compute metrics."""
        save_name = pdb_names[0].split(".")[0]
        frame_time = self._model_conf.frame_time
        sample_length = valid_feats["aatype"].shape[-1]
        diffuse_mask = np.ones(sample_length)
        b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

        # === Step 1. prepare init feats ===
        init_feats = self._prepare_init_feats(valid_feats, device, frame_time, sample_length)

        # === Step 2. inference ===
        sample_out = self.inference_fn(init_feats, num_t=num_t, min_t=min_t, aux_traj=True, noise_scale=noise_scale)

        # === Step 3. alignment ===
        align_sample, align_metric_list = self._align_predictions(sample_out["prot_traj"][0], valid_feats)

        # === Step 4. compute metrics ===
        result_metrics = self._compute_metrics(valid_feats, sample_out, align_sample, frame_time)

        # === Step 5. save predictions ===
        pdb_paths = self._save_eval_outputs(
            save_name, valid_feats, sample_out, align_sample, dirs, b_factors, is_training
        )

        # === Step 6. rotation / translation error ===
        tmp_ref_rigids = valid_feats["ref_rigids_0"][0, -1].cpu().numpy()
        rot_trans_err = self._calc_rot_trans_error(
            sample_out["rigid_traj"][0],
            gt_rigids=init_feats["rigids_0"].cpu().numpy(),
            ref_rigids=tmp_ref_rigids,
        )

        return {
            "save_name": save_name,
            "start_index": start_index,
            "metrics": result_metrics,
            "pdb_paths": pdb_paths,
            "rot_trans_err": rot_trans_err,
        }
    
    def _accumulate_metrics(self, metrics, result):
        """Accumulate results for later aggregation."""
        metrics["save_name_list"].append(result["save_name"])
        metrics["start_index_list"].append(result["start_index"])
        metrics["metric_list"].append(result["metrics"]["mean_eval"])
        metrics["metric_all_list"].append(result["metrics"]["all_eval"])
        metrics["metric_aligned_list"].append(result["metrics"]["align_mean_eval"])
        metrics["metric_aligned_all_list"].append(result["metrics"]["align_all_eval"])
        metrics["first_frame_all_list"].append(result["metrics"]["first_frame_eval"])
        metrics["save_pdb_dict"][result["save_name"]] = result["pdb_paths"]

        ave_quat, ave_trans, ref_ave_quat, ref_ave_trans, time_rot_dif, time_trans_dif = result["rot_trans_err"]
        rdict = metrics["rot_trans_error_dict"]
        rdict["name"].append(result["save_name"])
        rdict["ave_rot"].append(ave_quat)
        rdict["ave_trans"].append(ave_trans)
        rdict["first_rot"].append(ref_ave_quat)
        rdict["first_trans"].append(ref_ave_trans)
        rdict["time_rot_dif"].append(time_rot_dif)
        rdict["time_trans_dif"].append(time_trans_dif)

    def _finalize_eval_outputs(self, metrics, dirs, eval_dir):
        """Aggregate metrics, plot results, and return evaluation summary."""
        # === change the DataFrame ===
        ckpt_eval_metrics = pd.DataFrame(metrics["metric_list"])
        ckpt_eval_metrics.insert(0, "pdb_name", metrics["save_name_list"])
        ckpt_eval_metrics.to_csv(os.path.join(eval_dir, "metrics.csv"), index=False)

        ckpt_eval_metrics_aligned = pd.DataFrame(metrics["metric_aligned_list"])
        ckpt_eval_metrics_aligned.insert(0, "pdb_name", metrics["save_name_list"])
        ckpt_eval_metrics_aligned.to_csv(os.path.join(eval_dir, "metrics_aligned.csv"), index=False)

        RefAsInfer_DF = pd.DataFrame(metrics["first_frame_all_list"])
        RefAsInfer_DF.insert(0, "pdb_name", metrics["save_name_list"])
        # === Comparison Curver ===
        metric_merge_dict = {
            "Pred": ckpt_eval_metrics,
            "RefAsInfer": RefAsInfer_DF,
        }
        
        curve_fig = plot_curve_merged(metric_merge_dict, eval_dir, row_num=3, col_num=len(metrics["save_name_list"]))
        curve_fig_aligned = plot_curve_merged(metric_merge_dict, eval_dir, row_num=3, col_num=len(metrics["save_name_list"]), suffer_fix="aligned")
        error_fig = plot_rot_trans_curve(metrics["rot_trans_error_dict"], save_path=eval_dir, frame_step=self._data_conf.frame_sample_step)

        # === Get the Average Score ===
        ckpt_eval_metrics = ckpt_eval_metrics.applymap(
            lambda x: float(x) if isinstance(x, np.ndarray) else x
        )
        mean_dict = ckpt_eval_metrics.drop(columns=["pdb_name"]).mean().to_dict()
        rot_trans_error_mean = {k: np.mean(v) for k, v in metrics["rot_trans_error_dict"].items() if k != "name"}

        model_ckpt_update = self._update_best_model(mean_dict, rot_trans_error_mean)
            
        self._log_eval_summary(mean_dict, rot_trans_error_mean)

        return {
            "metrics": ckpt_eval_metrics,
            "fig_dis_curve": curve_fig,
            "fig_dis_curve_aligned": curve_fig_aligned,
            "fig_error": error_fig,
            "model_ckpt_update": model_ckpt_update,
            "rot_trans_error_mean": rot_trans_error_mean,
            "save_pdb_dict": metrics["save_pdb_dict"],
        }

    def _align_predictions(self, pred_traj, valid_feats):
        """
        Align each frame of predicted trajectory to the reference CA positions.
        Args:
            pred_traj: np.ndarray or torch.Tensor, shape [T, N, 37, 3]
            valid_feats: dict containing 'ref_atom37_pos' and 'atom37_mask'
        Returns:
            aligned_sample: torch.Tensor [T, N, 37, 3]
            align_metric_list: list of (rot, trans)
        """
        if torch.is_tensor(pred_traj):
            pred_traj = pred_traj.detach().cpu().numpy()

        ref_ca = valid_feats["ref_atom37_pos"][0][0].cpu().numpy()[:, 1]  # reference CA
        atom37_mask = valid_feats["atom37_mask"][0].cpu().numpy()

        align_sample_list, align_metric_list = [], []

        for frame_idx in range(pred_traj.shape[0]):
            sup = SVDSuperimposer()
            sup.set(ref_ca, pred_traj[frame_idx][:, 1])  # align on CA
            sup.run()
            rot, trans = sup.get_rotran()

            align_metric_list.append((rot, trans))
            aligned = np.dot(pred_traj[frame_idx], rot) + trans
            aligned *= atom37_mask[frame_idx][..., None]  # apply mask
            # align_sample_list.append(torch.from_numpy(aligned))
            align_sample_list.append(aligned)

        aligned_sample = torch.stack(align_sample_list)
        return aligned_sample, align_metric_list
    
    def _compute_metrics(self, valid_feats, sample_out, align_sample, frame_time):
        """
        Compute validation metrics for predicted, aligned, and reference trajectories.
        """
        gt_pos = valid_feats["atom37_pos"][0]
        gt_mask = valid_feats["atom37_mask"][0]
        ref_pos = valid_feats["ref_atom37_pos"][0, -1]  # last reference frame
        aatype = valid_feats["aatype"][0].cpu().numpy()

        # === Reference metrics (Ref as fake prediction)
        fake_res = torch.stack([ref_pos] * frame_time)
        first_eval_dic = compute_validation_metrics_all(
            gt_pos=gt_pos,
            out_pos=fake_res.cpu().numpy(),
            gt_mask=gt_mask,
            superimposition_metrics=True,
        )

        # === Unaligned prediction metrics
        eval_dic = compute_validation_metrics_all(
            gt_pos=gt_pos,
            out_pos=sample_out["prot_traj"][0],
            gt_mask=gt_mask,
            superimposition_metrics=True,
        )
        mean_eval_dic = {k: sum(v) / len(v) for k, v in eval_dic.items()}

        # === Aligned prediction metrics
        align_eval_dic = compute_validation_metrics_all(
            gt_pos=gt_pos,
            out_pos=align_sample.cpu().numpy(),
            gt_mask=gt_mask,
            superimposition_metrics=True,
        )
        align_mean_eval_dic = {k: sum(v) / len(v) for k, v in align_eval_dic.items()}

        return {
            "first_frame_eval": {
                k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in first_eval_dic.items()
            },
            "mean_eval": {
                k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in mean_eval_dic.items()
            },
            "all_eval": {
                k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in eval_dic.items()
            },
            "align_mean_eval": {
                k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in align_mean_eval_dic.items()
            },
            "align_all_eval": {
                k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in align_eval_dic.items()
            },
        }

    def _save_eval_outputs(
        self,
        save_name,
        valid_feats,
        sample_out,
        align_sample,
        dirs,
        b_factors,
        is_training=True,
    ):
        """
        Save GT, predicted, and aligned structures in PDB and NPZ format.
        """
        gt_path = os.path.join(dirs["gt"], f"{save_name}_gt.pdb")
        sample_path = os.path.join(dirs["sample"], f"{save_name}.pdb")
        sample_aligned_path = os.path.join(dirs["sample"], f"{save_name}_aligned.pdb")
        first_motion_path = os.path.join(dirs["sample"], f"{save_name}_first_motion.pdb")

        # === save GT
        aatype = valid_feats["aatype"][0, 0].cpu().numpy()
        
        # === save npz if not training
        if not is_training and dirs.get("pred_npz"):

            # also save "first" reference frame for visualization
            au.write_prot_to_pdb(
                prot_pos=valid_feats["ref_atom37_pos"][0].cpu().numpy(),
                file_path=os.path.join(dirs["sample"], f"{save_name}_first.pdb"),
                aatype=aatype,
                no_indexing=True,
                b_factors=b_factors,
            )

        return {"gt": gt_path, "gen": sample_path}

    def _prepare_extension_eval_dirs(self, eval_dir):
        pdb_base_path = os.path.join(eval_dir, "extension_pdb")
        ref_base_path = os.path.join(eval_dir, "reference_pdb")
        os.makedirs(pdb_base_path, exist_ok=True)
        os.makedirs(ref_base_path, exist_ok=True)
        return pdb_base_path, ref_base_path
    
    def _process_one_protein_extrapolation(
        self,
        extrapolation_time, 
        valid_feats,
        pdb_names,
        ref_base_path,
        pdb_base_path,
        device,
        min_t=None,
        num_t=None,
        noise_scale=1.0,
        executor=None,
    ):
        """Process one protein sequence and perform trajectory extrapolation."""
        # === Preparation ===
        protein_name = pdb_names[0]
        frame_time = self._model_conf.frame_time
        ref_number = self._model_conf.ref_number
        motion_number = self._model_conf.motion_number
        aatype = valid_feats["aatype"].cpu().numpy()
        sample_length = aatype.shape[-1]
        b_factors = np.tile((np.ones(sample_length) * 100)[:, None], (1, 37))
        pdb_path = os.path.join(pdb_base_path, f"{protein_name}_time_{extrapolation_time}.pdb")
        if os.path.exists(pdb_path):
            print(f"✅ {protein_name} already existed in: {pdb_path}")
            return 
        
        # Save reference structure
        ref_all_atom_positions = valid_feats["ref_atom37_pos"][0].cpu().numpy()
        au.write_prot_to_pdb(
            prot_pos=ref_all_atom_positions,
            file_path=os.path.join(ref_base_path, f"{protein_name}.pdb"),
            aatype=aatype[0, 0],
            no_indexing=True,
            b_factors=b_factors,
        )
        print(f"[Eval] Processing {protein_name}, length={extrapolation_time}")
        # === Initialize input ===
        atom_traj, rigid_traj = [], []
        valid_feats = self._prepare_init_feats(valid_feats, device, frame_time, sample_length)
        
        frame_time, L = valid_feats['res_mask'].shape

        # pre-compute
        all_start_rigids = self.diffuser.sample_ref(
                n_samples=extrapolation_time * frame_time * L,
                as_tensor_7=True,
            )['rigids_t'].reshape(extrapolation_time, frame_time, L, 7).to(device)
        reverse_steps = self.reverse_steps         
        z_rot_all = torch.randn(len(reverse_steps),frame_time,L, 3,device=device)
        z_trans_all = torch.randn(len(reverse_steps),frame_time,L, 3,device=device)
        
        # # === Iterative inference ===
        pbar = tqdm(range(extrapolation_time), desc=f"{protein_name}", ncols=80)
        
        for j in pbar:
            # === Perform inference ===
            sample_out = self.inference_fn(
                valid_feats,
                num_t=num_t,
                min_t=min_t,
                aux_traj=True,
                noise_scale=noise_scale, 
                z_rot_all=z_rot_all,
                z_trans_all=z_trans_all,
            )
            atom_pred = sample_out["prot_traj"][0]
            rigid_pred = sample_out["rigid_traj"][0]
            # Save the results
            atom_traj.append(atom_pred[-frame_time:])
            rigid_traj.append(rigid_pred[-frame_time:])
            # === Update reference state ===
            valid_feats['rigids_t'] = all_start_rigids[j]
            valid_feats = self._update_ref_with_prediction(
                valid_feats,
                atom_pred,
                rigid_pred,
                ref_number,
                motion_number,
                device,
            )
        # === Concatenate trajectory and save ===
        atom_traj = torch.cat(atom_traj, dim=0)
        rigid_traj = torch.cat(rigid_traj, dim=0)
        if torch.is_tensor(atom_traj):
            atom_traj = atom_traj.detach().cpu().numpy()
        if torch.is_tensor(rigid_traj):
            rigid_traj = rigid_traj.detach().cpu().numpy()
        if executor is not None:
            executor.submit(
                au.write_prot_to_pdb, 
                prot_pos=atom_traj,
                file_path=pdb_path,
                aatype=aatype[0, 0],
                no_indexing=True,
                b_factors=b_factors
            )
        else:
            au.write_prot_to_pdb(
                prot_pos=atom_traj,
                file_path=pdb_path,
                aatype=aatype[0, 0],
                no_indexing=True,
                b_factors=b_factors
            )
        
    def _prepare_init_feats(self, valid_feats, device, frame_time, sample_length):
        """Prepare initial features for inference."""
        res_mask = np.ones((frame_time, sample_length))
        fixed_mask = np.zeros_like(res_mask)
        res_idx = torch.arange(1, sample_length + 1).unsqueeze(0).repeat(frame_time, 1)

        # sample reference rigids_t
        ref_sample = self.diffuser.sample_ref(
            n_samples=sample_length * frame_time,
            as_tensor_7=True,
        )
        ref_sample["rigids_t"] = ref_sample["rigids_t"].reshape([-1, frame_time, sample_length, 7])
        ref_sample = tree.map_structure(lambda x: x.to(device), ref_sample)

        # compact input
        init_feats = {
            "res_mask": torch.tensor(res_mask[None], device=device),
            "seq_idx": res_idx[None].to(device),
            "fixed_mask": torch.tensor(fixed_mask[None], device=device),
            "torsion_angles_sin_cos": torch.zeros((1, sample_length, 7, 2), device=device),
            "sc_ca_t": torch.zeros((1, frame_time, sample_length, 3), device=device),
            **{k: v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device)
            for k, v in valid_feats.items()},
            **ref_sample,
        }

        # flatten: batch and temporal
        for key, value in init_feats.items():
            if key not in ["t"]: # [B,F,....]
                init_feats[key] = value.flatten(0, 1)
                # current we also flatten the node and edge repr ["node_repr", "edge_repr"]
                # from [B, N, D] & [B, N, N, D] -> [B*N, D] & [B*N, N, D] as B=1
                # TODO maybe should support B!=1

        return init_feats
    
    def _update_ref_with_prediction(self, valid_feats, atom_pred, rigid_pred, ref_number, motion_number, device, change_ref=False):
        concat_rigids = torch.cat([
            valid_feats["motion_rigids_0"],
            valid_feats["ref_rigids_0"],
            rigid_pred
        ], dim=0)

        concat_atoms = torch.cat([
            valid_feats["motion_atom37_pos"],
            valid_feats["ref_atom37_pos"],
            atom_pred
        ], dim=0)

        valid_feats["motion_rigids_0"] = concat_rigids[-motion_number:]
        valid_feats["motion_atom37_pos"] = concat_atoms[-motion_number:]
        
        return valid_feats
    
    def _update_best_model(self, mean_dict: dict, rot_trans_error_mean: dict) -> bool:
        """
        Compare current evaluation metrics with previous best; update if improved.
        """
        model_ckpt_update = False

        # Determine if improved (thresholds can be adjusted as needed)
        better_rmse = mean_dict["rmse_all"] < self.best_rmse_all
        better_rot = rot_trans_error_mean["ave_rot"] < self.best_rot_error
        better_trans = rot_trans_error_mean["ave_trans"] < self.best_trans_error

        if better_rmse or better_rot or better_trans:
            self.best_rmse_all = mean_dict["rmse_all"]
            self.best_rmse_ca = mean_dict["rmse_ca"]
            self.best_drmsd = mean_dict["drmsd_ca"]
            self.best_rmsd_ca_aligned = mean_dict["rmsd_ca_aligned"]

            self.best_rot_error = rot_trans_error_mean["ave_rot"]
            self.best_trans_error = rot_trans_error_mean["ave_trans"]
            self.best_ref_rot_error = rot_trans_error_mean["first_rot"]
            self.best_ref_trans_error = rot_trans_error_mean["first_trans"]

            self.best_trained_steps = self.trained_steps
            self.best_trained_epoch = self.trained_epochs
            model_ckpt_update = True

        return model_ckpt_update
    def _log_eval_summary(self, mean_dict: dict, rot_trans_error_mean: dict) -> None:
        """
        Print evaluation summary and current best metrics.
        """
        info = f"Step:{self.trained_steps} "
        for k, v in mean_dict.items():
            info += f"avg_{k}:{v:.4f} "
        for k, v in rot_trans_error_mean.items():
            if k != "name":
                info += f"avg_{k}:{v:.4f} "

        self._log.info("Evaluation Results: " + info)
        self._log.info(
            f"Best so far | steps/epoch: {self.best_trained_steps}/{self.best_trained_epoch} | "
            f"rmse_all: {self.best_rmse_all:.4f} | "
            f"rmse_ca: {self.best_rmse_ca:.4f} | "
            f"rmsd_ca_aligned: {self.best_rmsd_ca_aligned:.4f} | "
            f"drmsd_ca: {self.best_drmsd:.4f} | "
            f"rot_error: {self.best_rot_error:.4f}/{self.best_ref_rot_error:.4f} | "
            f"trans_error: {self.best_trans_error:.4f}/{self.best_ref_trans_error:.4f}"
        )

@hydra.main(version_base=None, config_path="./config", config_name="train_DyneTrion")
def run(conf: DictConfig) -> None:

    exp = Experiment(conf=conf)
    exp.start_training()


if __name__ == '__main__':
    run()
