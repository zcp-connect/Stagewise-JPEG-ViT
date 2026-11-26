import torch
import torch.distributed as dist
import random
import numpy as np
import logging
import math
import os
import psutil
from collections.abc import Mapping

import utils.custom_optims as coptim
import models.plainvit as pvit
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import datasets as ds
from utils.mp_scripts import ImageResizer
from torchinfo import summary

from time import time


def move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    if isinstance(data, Mapping):
        return {key: move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    if isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    return data


class LocalWindow():
    """
    Calculates loss in the given window
    Inputs:
        maxsize: size of the window to calculate windowed loss
    """
    def __init__(self, maxsize):
        self.queue = []
        self.maxsize = maxsize
    
    def put(self, val, returnval=False):
        """
        Enqueues loss into `maxsize` queue.
        Input:
            val: value of the loss
            returnval: returns loss if true
        """
        self.queue.append(val)
        if len(self.queue) > self.maxsize:
            self.queue.pop(0)
        if returnval:
            return self.calc_loss()
    
    def calc_loss(self):
        """
        Calculates loss in queue
        """
        assert len(self.queue) > 0, "List cannot be length 0"
        avg = sum(self.queue) / len(self.queue)
        return avg

def unpack_data(data, dataset, avail_device, mixup=None, nomixup=False):
    """
    unpacks data appropriately based on dataset.
    Args:
        data: data retrieved from dataloader
        dataset: string containing the type of dataset the model's training on
        avail_device: rank of a device or 'cuda'
        mixup: optional, apply mixup if given
        nomixup: if True, don't use mixup
    """
    inputs, labels = data
    _ = ds.dataset_to_stage(dataset)
    inputs = move_to_device(inputs, avail_device)
    labels = labels.to(avail_device, non_blocking=True)
    return inputs, labels

def dist_setup(rank:int, world_size:int, port:int=13933):
    """
    Setup for distributed data parallel

    rank: index of the process group (# of process group = # of gpus)
    world_size: number of processes in the group (# of gpus)
    port: port for DDP communication
    """
    dist.init_process_group("nccl", init_method=f"tcp://localhost:{port}",
                            rank=rank, world_size=world_size)
    print(f"Process {rank+1}/{world_size} initialized")

def adjust_lr(optimizer:torch.optim.Optimizer, lr):
    """
    changes the learning rate of optimizer to lr
    """
    for g in optimizer.param_groups:
        g['lr'] = lr

def copy_lr(optim_src:torch.optim.Optimizer, optim_dst:torch.optim.Optimizer):
    """
    Copies learning rate from optim_src to optim_dst
    """
    src_lr = optim_src.param_groups[0]['lr']
    for dst_g in optim_dst.param_groups:
        dst_g['lr'] = src_lr

def set_seeds(seed:int=11997733):
    """
    Set seeds for python, pytorch, and numpy. Default seed=11997733.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_threads(num_cpus, world_size, rank):
    """
    Returns appropriate number of threads for given num_cpus and world_size.
    Gives warning if the setting is inefficient

    Input:
        num_cpus: number of total CPU threads
        world_size: number of GPU devices
        rank: GPU index. Used for logging
    Returns:
        threads: number of threads per GPU
    """
    threads = num_cpus // world_size
    if rank == 0 and num_cpus % world_size != 0:
        logging.warning(f"Total number of CPUs ({num_cpus}) is not an integer multiple of number of GPUs ({world_size}). This will lead to inefficient CPU utilization. ({threads} CPUs per GPU)")
    return threads

def update_config(cfg, rank, world_size, threads, run_train, run_eval, deterministic):
    """
    Updates config appropriately given dataset

    Input:
        cfg: yacs config node
        rank: GPU index rank
        world_size: number of available GPUs
        threads: number of threads per GPU
        run_train, run_eval: If true, run train or eval pipelie
        deterministic: If True, set deterministic mode (slows down pipeline but provides more-or-less reproducible results)
    """
    cfg.RANK=rank
    cfg.WORLDSIZE=world_size
    cfg.THREADS=threads
    grad_accum_steps = max(1, getattr(cfg.TRAIN, "GRADACCUMSTEPS", 1))
    denom = world_size * grad_accum_steps
    assert cfg.TRAIN.BATCHSIZE % denom == 0, (
        f"Overall batch size ({cfg.TRAIN.BATCHSIZE}) must be divisible by "
        f"world_size * grad_accum_steps ({world_size} * {grad_accum_steps} = {denom})"
    )
    cfg.TRAIN.BATCHPERGPU = cfg.TRAIN.BATCHSIZE // denom # micro-batch size per GPU
    cfg.TRAIN.RUNTRAIN=run_train
    cfg.TRAIN.RUNEVAL=run_eval
    cfg.TRAIN.DETERMINISTIC=deterministic
    stage = ds.normalize_stage_name(cfg.MODEL.STAGE)
    cfg.MODEL.STAGE = stage
    cfg.TRAIN.DATASET = ds.stage_to_dataset(stage)
    rlc_cp_direct_stages = getattr(ds, "RLC_CP_DIRECT_STAGES", set())
    huffman_cp_direct_stages = getattr(ds, "HUFFMAN_CP_DIRECT_STAGES", set())
    if stage in ("rgb", "ycbcr"):
        cfg.MODEL.INPUTSIZE = (3, 224, 224)
    elif stage in getattr(ds, "GROUP_DCT_STAGES", set()):
        cfg.MODEL.INPUTSIZE = (1, 28, 28, 8, 8)
    elif stage in ds.NUMERIC_DCT_STAGES:
        cfg.MODEL.INPUTSIZE = (196, cfg.MODEL.NUMERIC_FEATURE_DIM)
    elif stage in ("raw_byte", "bitstream"):
        cfg.MODEL.INPUTSIZE = (196, cfg.MODEL.RLC_LMAX)  # per-MCU byte length follows --rlc_lmax (default 320)
    elif stage in (
        "rlc_dc", "rlc_dcac", "rlc_dc_simple", "rlc_dcac_simple",
        "rlc_dc_attnpool", "rlc_dcac_attnpool",
        "rlc_dc_2d", "rlc_dcac_2d", "rlc_dc_2d_max", "rlc_dcac_2d_max",
        "rlc_dc_flat", "rlc_dcac_flat",
        "rlc_dc_2d_cp", "rlc_dcac_2d_cp",
        "rlc_dc_2d_direct", "rlc_dcac_2d_direct",
        "rlc_dc_2d_max_direct", "rlc_dcac_2d_max_direct",
        "rlc_dc_flat_direct", "rlc_dcac_flat_direct",
        "rlc_dc_2d_cp_direct", "rlc_dcac_2d_cp_direct",
        "huff_dcac_2d_direct", "huff_dcac_flat_direct", "event_amp_only_flat_direct", "huff_dcac_2d_cp_direct",
        *rlc_cp_direct_stages,
        *huffman_cp_direct_stages,
    ):
        rlc_lmax = 8 if stage in (
            "rlc_dc", "rlc_dc_simple", "rlc_dc_attnpool",
            "rlc_dc_2d", "rlc_dc_2d_max", "rlc_dc_flat", "rlc_dc_2d_cp",
            "rlc_dc_2d_direct", "rlc_dc_2d_max_direct",
            "rlc_dc_flat_direct", "rlc_dc_2d_cp_direct",
        ) else cfg.MODEL.RLC_LMAX
        cfg.MODEL.INPUTSIZE = (196, rlc_lmax)
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    cfg.MODEL.SUMDTYPE=['fp32']
    return cfg

def get_mixup(cfg):
    """
    returns appropriate mixup for given config
    
    Input:
        cfg: yacs config node
    Returns:
        Mixup
    """
    return None

def get_deletion_path(delete_dataset, temp_datapath):
    """
    Returns appropriate deletion path given delete_dataset flag

    Input:
        delete_dataset: If True, delete dataset after training
        temp_datapath: If delete_dataset is true, delete this temporary datapath after training
    Returns:
        deletion path
    """
    path_to_delete = None
    if delete_dataset:
        path_to_delete = temp_datapath
        logging.warning(f"\n===== Dataset will be DELETED after train/eval =====\n")
        logging.warning(f"==== Deletion Path: {path_to_delete}")
    return path_to_delete

def copy_dataset(datapath, temp_datapath, indexpaths, num_cpus, no_extract, no_resize, use_msrsync, verbose):
    """
    Copies, extracts, and resizes the dataset

    Inputs:
        datapath: path containing .tar and .sh files
        temp_datapath: path to copy/extract .tar files
        indexpaths: list containing index file paths. [0] = train, [1] = val
        num_cpus: number of total available cpu threads
        no_extract: If true, assume .tar is already extracted to temp_datapath
        no_resize: If set, assume .tar is already extracted to temp_datapath AND resized to 512x512
        use_msrsync: If set, use msrsync to directly copy the images from datapath to temp_datapath (need to download msrsync: https://github.com/jbd/msrsync)
        verbose: If true, report progress
    """
    p = psutil.Process() # Force Python to utilize all CPUs if possible
    p.cpu_affinity(list(range(psutil.cpu_count())))

    if use_msrsync:
        # copy `datapath` to `temp_datapath`. Assume `datapath` contains already-extracted and processed images
        logging.info("Copying ImageNet train")
        ds.run_msrsync( # train
            msrsync_path='~/msrsync', 
            source=os.path.join(datapath, 'train'), 
            dest=temp_datapath, # if using `tmpprefix`, this should be <temp_datapath>/<tmpprefix>
            bucketpath=os.path.join(temp_datapath, "msrsync_temp"),
            process=num_cpus, verbose=True
            )
        logging.info("Copying ImageNet val")
        ds.run_msrsync( # val
            msrsync_path='~/msrsync', 
            source=os.path.join(datapath, 'val'), 
            dest=temp_datapath, # if using `tmpprefix`, this should be <temp_datapath>/<tmpprefix>
            bucketpath=os.path.join(temp_datapath, "msrsync_temp"),
            process=num_cpus, verbose=True
            )
    else:
        if not no_extract:
            tarhandler = ds.imagenet_tar_handler(
                tarpath=datapath, tmppath=temp_datapath, tmpprefix="", num_proc=num_cpus, verbose=verbose
            )
            assert tarhandler.checkfile() == True, "Tarhandler failed to verify .tar and .sh files at datapath."
            tarhandler.tar2tmp(copy=False) # don't copy tar files to tmp datapath (if .tar files are on network file system, it may be better to copy it to local storage. In such cases, change the flag to True)

        basepath = temp_datapath # if using `tmpprefix`, this should be <temp_datapath>/<tmpprefix>
        if not no_resize:
            logging.info("Resizing extracted data to 512x512 (Takes ~30 min with Intel Xeon Gold w/ 32 threads)")
            train_resizer = ImageResizer(512, num_cpus, indexpaths[0], basepath, basepath, verbose=verbose)
            train_resizer.start_processes()
            train_resizer.wait_and_terminate()
            val_resizer = ImageResizer(512, num_cpus, indexpaths[1], basepath, basepath, verbose=verbose)
            val_resizer.start_processes()
            val_resizer.wait_and_terminate()
        else:
            # If no resize, make sure one PNG file in the dataset is converted to JPEG
            transcode_path = os.path.join(basepath, 'train/n02105855/n02105855_2933.JPEG')
            try:
                ds.transcode_to_jpeg(filepath=transcode_path)
            except Exception as e:
                logging.warning(f"Failed to transcode \"{transcode_path}\" to JPEG. This file is encoded in PNG -- this can cause issues for DCT/JPEG models. Raw error message: {e}\n===============")

def get_dataset(cfg, temp_datapath, indexpaths):
    """
    Generate dataloaders from the given dataset

    Inputs:
        cfg: yacs config node
        temp_datapath: path to copy/extract .tar files
        indexpaths: list containing index file paths. [0] = train, [1] = val
    """
    basepath = temp_datapath # if using `tmpprefix`, this should be <temp_datapath>/<tmpprefix>

    if cfg.TRAIN.SPLIT > 0: # train / minipatch dataset split
        trainloader, valloader, trainvalloader = ds.dataset_selector(
            dataset=cfg.TRAIN.DATASET, type='train', indexpath=indexpaths[0], basepath=basepath,
            batch_size=cfg.TRAIN.BATCHPERGPU, num_workers=cfg.THREADS, shuffle=True, trainval_split=cfg.TRAIN.SPLIT,
            return_indices=False, distributed=True, rank=cfg.RANK, world_size=cfg.WORLDSIZE, seed=cfg.SEED,
            ops_list=cfg.TRAIN.AUGLIST, num_ops=cfg.TRAIN.NUMOPS, ops_magnitude=cfg.TRAIN.AUGSTR,
            freqnorm=cfg.TRAIN.FREQNORM, freqnorm_stats=cfg.TRAIN.FREQNORM_STATS,
            rlc_lmax=cfg.MODEL.RLC_LMAX,
            prefetch_factor=cfg.TRAIN.PREFETCH_FACTOR,
            in_order=cfg.TRAIN.IN_ORDER,
        )

    testloader = ds.dataset_selector(
        dataset=cfg.TRAIN.DATASET, type='test', indexpath=indexpaths[1], basepath=basepath,
        batch_size=cfg.TRAIN.BATCHPERGPU, num_workers=cfg.THREADS, shuffle=False, 
        distributed=True, rank=cfg.RANK, world_size=cfg.WORLDSIZE, seed=cfg.SEED,
        freqnorm=cfg.TRAIN.FREQNORM, freqnorm_stats=cfg.TRAIN.FREQNORM_STATS,
        rlc_lmax=cfg.MODEL.RLC_LMAX,
        prefetch_factor=cfg.TRAIN.PREFETCH_FACTOR,
        in_order=cfg.TRAIN.IN_ORDER)
    
    return trainloader, valloader, trainvalloader, testloader

def set_device(cfg, verbose=False):
    """
    Sets cuda device and empty GPU cache in preparation for pipieline.
    Additionally sets deterministic mode if enabled.

    Inputs:
        cfg: yacs config node
        verbose: If True, report progress
    """
    if verbose:
        logging.info(f"Process {cfg.RANK+1}/{cfg.WORLDSIZE} set device and emptying cache")
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available; skipping CUDA device setup.")
        return
    torch.cuda.set_device(cfg.RANK)
    torch.cuda.empty_cache()
    if cfg.TRAIN.DETERMINISTIC:
        torch.backends.cudnn.deterministic = True # deterministic behavior for cudnn
        torch.backends.cudnn.benchmark = False # Benchmark off -- chooses default algorithm, not the best algorithm through benchmarking
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8" # look https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility at 2.1.4. Results reproducibility for more detail
        torch.use_deterministic_algorithms(mode=True, warn_only=True) # Warn only so that even if you use non-deterministic layers, it doesn't raise an error

def get_dtype(dtypestring):
    """
    Converts `dtypestring' to torch dtype

    Inputs:
        dtypestring: string containing dtype (one of fp16, fp32, bf16)
    Outputs:
        pytorch dtype
    """
    # get dtype
    if dtypestring=='fp32':
        dtype = torch.float32
    elif dtypestring=='fp16':
        dtype = torch.float16
    elif dtypestring=='bf16':
        dtype = torch.bfloat16
    else:
        logging.error(f"Unsupported datatype: {dtypestring} (currently supported: fp16, fp32, bf16)")
    return dtype

def get_model(cfg, report=True):
    """
    Returns appropriate ViT model given input config
    Inputs:
        cfg: yacs config node
    Returns:
        ViT Model (pytorch model)
    """
    dtype = get_dtype(cfg.MODEL.DTYPE)
    model_device = cfg.RANK if torch.cuda.is_available() else "cpu"
    vitmodel = pvit.ViT(
        in_channels=3,
        patch_size=cfg.MODEL.PATCHSIZE,
        emb_size=cfg.MODEL.EMBEDSIZE,
        depth=cfg.MODEL.DEPTH,
        n_classes=cfg.MODEL.CLASSES,
        drop_p=cfg.TRAIN.DROP,
        device=model_device,
        dtype=dtype,
        num_heads=cfg.MODEL.HEADS,
        head_size=cfg.MODEL.HEADSIZE,
        pixel_space=cfg.MODEL.DOMAIN,
        ver=cfg.MODEL.VERSION,
        use_subblock=cfg.MODEL.SUBBLOCK,
        stage=cfg.MODEL.STAGE,
        numeric_feature_dim=cfg.MODEL.NUMERIC_FEATURE_DIM,
        rlc_lmax=cfg.MODEL.RLC_LMAX,
    )

    if cfg.RANK==0 and report:
        log_configs(cfg, vitmodel)
        
    return vitmodel

def log_configs(cfg, vitmodel):
    """
    Report hyperparameters of input config to logger

    Input:
        cfg: yacs config node
        vitmodel: ViT model (PyTorch)
    """
    summary_dtype = [get_dtype(item) for item in cfg.MODEL.SUMDTYPE]
    try:
        summary(vitmodel.to(cfg.RANK), input_data=make_synthetic_stage_input(cfg, batch_size=1), col_names=cfg.MODEL.SUMCOL, device=cfg.RANK, verbose=1)
    except Exception as exc:
        logging.warning(f"torchinfo summary skipped for stage={cfg.MODEL.STAGE}: {exc}")

    logging.info(f"Dataset: {cfg.TRAIN.DATASET}")
    logging.info(f"Stage: {cfg.MODEL.STAGE}, embedding version: {cfg.MODEL.VERSION}, dtype: {cfg.MODEL.DTYPE}")
    logging.info(f"Numeric feature dim: {cfg.MODEL.NUMERIC_FEATURE_DIM}, RLC lmax: {cfg.MODEL.RLC_LMAX}")
    logging.info(f"Hyperparams: ep: {cfg.TRAIN.EPOCHS}, lr: {cfg.TRAIN.LR:.2e}, wd: {cfg.TRAIN.WD:.2e}, drop: {cfg.TRAIN.DROP}, batchsize(/gpu): {cfg.TRAIN.BATCHPERGPU}, grad_accum: {cfg.TRAIN.GRADACCUMSTEPS}")
    logging.info(f"AMP mode: {cfg.TRAIN.AMP}, AMP dtype: {cfg.MODEL.AMPDTYPE}")
    logging.info(f"Seed: {cfg.SEED}")
    logging.info(f"Deterministic mode: {cfg.TRAIN.DETERMINISTIC}")
    logging.info(f"RandAugment operations: {cfg.TRAIN.AUGLIST}")
    logging.info(f"Number of operations: {cfg.TRAIN.NUMOPS}")
    logging.info(f"RandAugment magnitude: {cfg.TRAIN.AUGSTR}")
    logging.info(f"Frequency normalization: {cfg.TRAIN.FREQNORM}, stats: {cfg.TRAIN.FREQNORM_STATS}")


def make_synthetic_stage_input(cfg, batch_size=None, device=None):
    batch_size = batch_size or cfg.TRAIN.BATCHPERGPU
    if device is None:
        device = cfg.RANK if torch.cuda.is_available() else "cpu"
    stage = ds.normalize_stage_name(cfg.MODEL.STAGE)
    if stage in ("rgb", "ycbcr"):
        return torch.randn(batch_size, 3, 224, 224, device=device)
    if stage in getattr(ds, "GROUP_DCT_STAGES", set()):
        y = torch.randn(batch_size, 1, 28, 28, 8, 8, device=device)
        cbcr = torch.randn(batch_size, 2, 14, 14, 8, 8, device=device)
        return {"y": y, "cbcr": cbcr}
    if stage in ds.NUMERIC_DCT_STAGES:
        return torch.randn(batch_size, 196, cfg.MODEL.NUMERIC_FEATURE_DIM, device=device)
    if stage in ("raw_byte", "bitstream"):
        payload = torch.randint(0, 256, (batch_size, 196, cfg.MODEL.RLC_LMAX), device=device, dtype=torch.uint8)
        mask = torch.ones((batch_size, 196, cfg.MODEL.RLC_LMAX), device=device, dtype=torch.bool)
        return payload, mask
    if stage in (
        "rlc_dc", "rlc_dcac", "rlc_dc_simple", "rlc_dcac_simple",
        "rlc_dc_attnpool", "rlc_dcac_attnpool",
        "rlc_dc_2d", "rlc_dcac_2d", "rlc_dc_2d_max", "rlc_dcac_2d_max",
        "rlc_dc_flat", "rlc_dcac_flat",
        "rlc_dc_2d_cp", "rlc_dcac_2d_cp",
        "rlc_dc_2d_direct", "rlc_dcac_2d_direct",
        "rlc_dc_2d_max_direct", "rlc_dcac_2d_max_direct",
        "rlc_dc_flat_direct", "rlc_dcac_flat_direct",
        "rlc_dc_2d_cp_direct", "rlc_dcac_2d_cp_direct",
        "huff_dcac_2d_direct", "huff_dcac_flat_direct", "event_amp_only_flat_direct", "huff_dcac_2d_cp_direct",
        *getattr(ds, "RLC_CP_DIRECT_STAGES", set()),
        *getattr(ds, "HUFFMAN_CP_DIRECT_STAGES", set()),
    ):
        lmax = 8 if stage in (
            "rlc_dc", "rlc_dc_simple", "rlc_dc_attnpool",
            "rlc_dc_2d", "rlc_dc_2d_max", "rlc_dc_flat", "rlc_dc_2d_cp",
            "rlc_dc_2d_direct", "rlc_dc_2d_max_direct",
            "rlc_dc_flat_direct", "rlc_dc_2d_cp_direct",
        ) else cfg.MODEL.RLC_LMAX
        if stage in (
            "rlc_dc_2d", "rlc_dcac_2d", "rlc_dc_2d_max", "rlc_dcac_2d_max",
            "rlc_dc_flat", "rlc_dcac_flat", "rlc_dc_2d_cp", "rlc_dcac_2d_cp",
            "rlc_dc_2d_direct", "rlc_dcac_2d_direct",
            "rlc_dc_2d_max_direct", "rlc_dcac_2d_max_direct",
            "rlc_dc_flat_direct", "rlc_dcac_flat_direct",
            "rlc_dc_2d_cp_direct", "rlc_dcac_2d_cp_direct",
            "huff_dcac_2d_direct", "huff_dcac_flat_direct", "event_amp_only_flat_direct", "huff_dcac_2d_cp_direct",
            *getattr(ds, "RLC_CP_DIRECT_STAGES", set()),
            *getattr(ds, "HUFFMAN_CP_DIRECT_STAGES", set()),
        ):
            if stage in getattr(ds, "HUFFMAN_CP_DIRECT_STAGES", set()):
                run = torch.randint(0, 1025, (batch_size, 196, lmax), device=device, dtype=torch.long)
            elif stage in ("huff_dcac_2d_direct", "huff_dcac_flat_direct", "event_amp_only_flat_direct"):
                run = torch.rand(batch_size, 196, lmax, device=device)
            else:
                run_dtype = torch.long if stage in (
                    "rlc_dc_2d_cp", "rlc_dcac_2d_cp",
                    "rlc_dc_2d_cp_direct", "rlc_dcac_2d_cp_direct",
                    *getattr(ds, "RLC_CP_DIRECT_STAGES", set()),
                ) else torch.float32
                run = torch.randint(0, 16, (batch_size, 196, lmax), device=device, dtype=run_dtype) if run_dtype is torch.long else torch.rand(batch_size, 196, lmax, device=device)
            value = torch.randn(batch_size, 196, lmax, device=device) * 0.5
            mask = torch.ones((batch_size, 196, lmax), device=device, dtype=torch.bool)
            return run, value, mask
        kind = torch.randint(0, 5, (batch_size, 196, lmax), device=device, dtype=torch.long)
        component = torch.randint(0, 4, (batch_size, 196, lmax), device=device, dtype=torch.long)
        block = torch.randint(0, 7, (batch_size, 196, lmax), device=device, dtype=torch.long)
        zigzag = torch.randint(0, 65, (batch_size, 196, lmax), device=device, dtype=torch.long)
        run = torch.randint(0, 17, (batch_size, 196, lmax), device=device, dtype=torch.long)
        size = torch.randint(0, 17, (batch_size, 196, lmax), device=device, dtype=torch.long)
        amplitude = torch.randn(batch_size, 196, lmax, device=device)
        mask = torch.ones((batch_size, 196, lmax), device=device, dtype=torch.bool)
        return kind, component, block, zigzag, run, size, amplitude, mask
    raise ValueError(f"Unsupported stage: {stage}")

def clip_gradscaler(gradscaler, scale_max=2**18, scale_min=2**(-4)):
    """
    Clips gradscaler scale such that it does not become exceedingly large.

    Input:
        gradscaler
    """
    if gradscaler._scale > scale_max: # maximum scale
        gradscaler._scale = torch.tensor(scale_max).to(gradscaler._scale)
    if gradscaler._scale < scale_min: # minimum scale
        gradscaler._scale = torch.tensor(scale_min).to(gradscaler._scale)

def get_tensorboard(cfg, savepath):
    """
    Returns appropriate tensorboard summarywriter for given config

    Inputs:
        cfg: yacs config node
        savepath: savepath of the model
    Returns
        SummaryWriter (tensorboard): wrtier for TensorBoard
    """
    tensorboard_path = os.path.join(os.path.dirname(savepath), "tensorboard")
    modelsavefile_name = os.path.basename(savepath)
    tensorboard_path = os.path.join(tensorboard_path, modelsavefile_name.rstrip('.pth')+"_lr{:.0E}_wd{:.0E}_drop{}".format(cfg.TRAIN.LR, cfg.TRAIN.WD, cfg.TRAIN.DROP))
    writer = SummaryWriter(log_dir=tensorboard_path, filename_suffix="_"+modelsavefile_name.rstrip('.pth'), )
    return writer

def write_tensorboard_train(writer, epoch, train_loss, val_loss, tval_loss, val_acc, tval_acc, lr):
    """
    Writes to tensorboard writer. To be used during training epochs

    Inputs:
        writer: Tensorboard writer
        train_loss: Training loss (train mode)
        val_loss: Validation loss
        tval_loss: Training loss (eval mode)
        val_acc: Validation Accuracy
        tval_acc: Training Accuracy (eval mode)
        lr: Learning rate
    """
    writer.add_scalar(tag='Loss/Train', scalar_value=train_loss, global_step = epoch+1)
    writer.add_scalar(tag='Loss/Val', scalar_value=val_loss, global_step = epoch+1)
    writer.add_scalar(tag='Loss/Train_val', scalar_value=tval_loss, global_step = epoch+1)
    writer.add_scalar(tag='Acc/Val', scalar_value=val_acc, global_step=epoch+1)
    writer.add_scalar(tag='Acc/Train_val', scalar_value=tval_acc, global_step=epoch+1)
    writer.add_scalar(tag='Learning Rate', scalar_value=lr, global_step=epoch+1)

def write_tensorboard_eval(
    writer, test_acc, test_loss, val_acc=None, val_loss=None, tval_acc=None, tval_loss=None,
    test_topk_acc=None, val_topk_acc=None, tval_topk_acc=None, top_k=5,
):
    """
    Writes to tensorboard writer. Used during evaluation (after training is complete)

    Inputs:
        writer: Tensorboard writer
        test_acc: Test Accuracy
        test_loss: Test Loss
        val_acc: Validation accuracy (if None, skips write)
        val_loss: Validation loss (if None, skips write)
        tval_acc: Train accuracy (eval mode) (if None, skips write)
        tval_loss: Train loss (eval mode) (if None, skips write)
        *_topk_acc: Top-K accuracy values (if None, skips write)
        top_k: K used for optional Top-K TensorBoard tags
    """
    writer.add_scalar(tag='Acc/Test', scalar_value=100 * test_acc, global_step=1)
    writer.add_scalar(tag='Loss/Test', scalar_value=test_loss, global_step=1)
    if test_topk_acc != None:
        writer.add_scalar(tag=f'Acc/TestTop{top_k}', scalar_value=100 * test_topk_acc, global_step=1)
    if val_acc != None:
        writer.add_scalar(tag='Acc/Val', scalar_value=100 * val_acc, global_step=1)
    if val_loss != None:
        writer.add_scalar(tag='Loss/Val', scalar_value=val_loss, global_step=1)
    if val_topk_acc != None:
        writer.add_scalar(tag=f'Acc/ValTop{top_k}', scalar_value=100 * val_topk_acc, global_step=1)
    if tval_acc != None:
        writer.add_scalar(tag='Acc/Train_val', scalar_value=100 * tval_acc, global_step=1)
    if tval_loss != None:
        writer.add_scalar(tag='Loss/Train_val', scalar_value=tval_loss, global_step=1)
    if tval_topk_acc != None:
        writer.add_scalar(tag=f'Acc/Train_valTop{top_k}', scalar_value=100 * tval_topk_acc, global_step=1)

def get_ckpt_path(savepath, epoch):
    """
    Returns appropriate checkpoint path in string. Create directory if it does not exist

    Inputs:
        savepath: savepath of the model
        epoch: current epoch of the model
    Returns
        checkpoint_path: string containing path to checkpoint
    """
    modelname = os.path.basename(savepath).split('.')[0]
    ckpt_savepath = os.path.join(os.path.dirname(savepath), 'checkpoints')
    ckpt_savepath = os.path.join(ckpt_savepath, modelname) # Base folder path. Will be joined with savefilename + epoch + .ckpt extension later
    if not os.path.exists(ckpt_savepath):
        os.makedirs(ckpt_savepath)
    ckptfilename = modelname + f"_{epoch+1}.ckpt"
    ckpt_savepath = os.path.join(ckpt_savepath, ckptfilename)
    return ckpt_savepath

def save_ckpt(ckptpath, epoch, itr, state_model, state_optim, state_wd, state_sched, state_scaler, loss, val_acc, verbose=False):
    """
    Save checkpoint

    Inputs:
        ckptpath: Path to save checkpoint
        epoch: current epoch (0-indexed)
        itr: current iteration
        state_model, state_optim, state_wd, state_sched, state_scaler: state_dict() of corresponding objects
        loss: training loss
        val_acc: validationa accuracy
        verbose: If true, report checkpoint save
    """
    torch.save({
        'epoch':(epoch+1),
        'current_itr':itr,
        'model_state_dict':state_model,
        'optimizer_state_dict':state_optim,
        'weight_decayer_state_dict':state_wd,
        'scheduler_state_dict':state_sched,
        "scaler_state_dict":state_scaler,
        'loss':loss,
        'val_accuracy':val_acc,
    }, ckptpath)

    if verbose:
        logging.info(f"Checkpoint saved at: {ckptpath}")

def get_optim_and_criterion(cfg, vitmodel, trainloader):
    """
    Returns optimizer and criterion for given config and vitmodel.

    Inputs:
        cfg: yacs config node
        vitmode: ViT model
        trainloader: dataloaders
    Returns:
        criterion: criterion for loss
        optimizer: parameter optimizer
        weight_decayer: weight decayer for weights, excluding biases
        cosinesceduler: learning rate cosine scheduler
        gradscaler: gradient scaler when using AMP, otherwise None
    """
    scheduler_epochs = cfg.TRAIN.SCHEDULER_EPOCHS if cfg.TRAIN.SCHEDULER_EPOCHS > 0 else cfg.TRAIN.EPOCHS
    grad_accum_steps = max(1, getattr(cfg.TRAIN, "GRADACCUMSTEPS", 1))
    steps_per_epoch = math.ceil(len(trainloader) / grad_accum_steps)
    maxiters = steps_per_epoch * scheduler_epochs

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(vitmodel.parameters(), lr=cfg.TRAIN.LR, weight_decay=0, eps=1e-8)
    weight_decayer = coptim.WeightDecay([param for name, param in vitmodel.named_parameters() if (".weight" in name) and ("lrnorm" not in name)], lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD) # handles weight decay part separately
    t_max = max(1, maxiters - cfg.TRAIN.WARMUP)
    cosinescheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0) # decrease every iteration
    gradscaler = None
    if cfg.TRAIN.AMP:
        gradscaler = torch.cuda.amp.GradScaler(growth_factor=cfg.TRAIN.SCALER.GROWTH, backoff_factor=cfg.TRAIN.SCALER.BACKOFF, growth_interval=cfg.TRAIN.SCALER.INTERVAL) # automatic mixed precision
    
    return criterion, optimizer, weight_decayer, cosinescheduler, gradscaler

def load_checkpoint(cfg, load_ckpt, vitmodel, optimizer, weight_decayer, cosinescheduler, gradscaler):
    """
    Loads checkpoint from load_ckpt
    Inputs:
        cfg: yacs config node
        load_ckpt: contains checkpoint path (if "": do not load checkpoint)
        vitmodel: ViT model (pytorch)
        optimizer: optimizer
        weight_decayer: weight_decayer
        cosinescheduler: cosine scheduler
        gradscaler: gradient scaler (if using AMP)
    Returns:
        Returns the following objects with checkpoint loaded
            ckpt_epoch: epoch from checkpoint
            vitmodel
            optimizer
            weight_decayer
            cosinescheduler
            gradscaler
    """
    ckpt_epoch=0
    current_itr=0
    if load_ckpt != "":
        map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.RANK} # mapping-based checkpoint loading for DDP
        checkpoint = torch.load(load_ckpt, map_location=map_location)

        ckpt_epoch = checkpoint['epoch']
        current_itr = checkpoint['current_itr']
        model_state_dict = checkpoint['model_state_dict']
        load_target = vitmodel
        if hasattr(vitmodel, 'module') and not next(iter(model_state_dict)).startswith('module.'):
            load_target = vitmodel.module
        load_target.load_state_dict(model_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        weight_decayer.load_state_dict(checkpoint['weight_decayer_state_dict'])
        cosinescheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if cfg.TRAIN.AMP:
            gradscaler.load_state_dict(checkpoint["scaler_state_dict"])
        if cfg.RANK==0:
            logging.info(f"----- Checkpoint loaded. Epoch/Iter: {ckpt_epoch}/{current_itr}, train_loss: {checkpoint['loss']:.4f}, val_acc: {checkpoint['val_accuracy']*100:.2f}% -----")
    return ckpt_epoch, current_itr, vitmodel, optimizer, weight_decayer, cosinescheduler, gradscaler

def load_model_from_ckpt(cfg, ckpt_path):
    """
    Loads checkpoint and extracts model state dict

    Inputs:
        cfg: yacs config node
        ckpt_path: checkpoint path
        outputpath: save model state dict to this path
    """
    map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.RANK} # mapping-based checkpoint loading for DDP
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    vitmodel = get_model(cfg, report=False)
    vitmodel = vitmodel.to(cfg.RANK)
    vitmodel.load_state_dict(checkpoint['model_state_dict'])
    vitmodel = DDP(vitmodel, device_ids=[cfg.RANK], output_device=cfg.RANK) # load model to correct devices
    if cfg.RANK==0:
        log_configs(cfg, vitmodel)
    return vitmodel

def load_model_and_report(cfg, savepath):
    """
    Loads model from savepath and return.
    If rank==0, report summary and hyperparameters of the loaded model.

    Inputs:
        cfg: yacs config node
        savepath: path to model
    """
    map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.RANK}
    vitmodel = get_model(cfg, report=False)
    vitmodel = vitmodel.to(cfg.RANK)
    vitmodel_state_dict = torch.load(savepath, map_location=map_location)
    vitmodel.load_state_dict(vitmodel_state_dict)
    vitmodel = DDP(vitmodel, device_ids=[cfg.RANK], output_device=cfg.RANK) # load model to correct devices
    if cfg.RANK==0:
        log_configs(cfg, vitmodel)
    return vitmodel
