import torch
import torch.distributed as dist
import torchmetrics
import logging
import utils.pipeline_utils as utils
import benchmark as bm

def evaluate_model(cfg, model, dataloader, criterion, verbose=False, top_k=1, return_metrics=False):
    """
    Evaluate model

    Inputs:
        cfg: yacs config node
        model: PyTorch model to evaluate
        dataloader: PyTorch dataloader
        crietrion: Criterion to evaluate on (ex. crossentropyloss)
        verbose: If true, report progress for rank 0 process (device_rank == 0)
    
    Outputs:
        Accuracy: Accuracy on dataset (float), or a metrics dict when return_metrics=True
        Loss: Loss on dataset (float)
    """
    # Read parameters from config
    dataset_name = cfg.TRAIN.DATASET
    device_rank = cfg.RANK
    use_amp = cfg.TRAIN.AMP
    num_classes = cfg.MODEL.CLASSES
    top_k = int(top_k)
    if top_k < 1 or top_k > num_classes:
        raise ValueError(f"top_k must be in [1, {num_classes}], got {top_k}")

    num_batch = len(dataloader)
    total_loss = torch.zeros(1, dtype=torch.float32, device=device_rank)
    acc_top1 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1, dist_sync_on_step=True, process_group=None).to(device_rank)
    acc_topk = None
    if top_k > 1:
        acc_topk = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=top_k, dist_sync_on_step=True, process_group=None).to(device_rank)
    model.eval()
    seen_batches = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            seen_batches += 1
            images, labels = utils.unpack_data(data, dataset_name, device_rank)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=utils.get_dtype(cfg.MODEL.AMPDTYPE)):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            acc_top1.forward(outputs, labels)
            if acc_topk is not None:
                acc_topk.forward(outputs, labels)
            if verbose and device_rank==0:
                print(f'\rEvaluating... {i+1}/{num_batch}   ', end="", flush=True)
    # average loss across GPUs
    if seen_batches == 0:
        raise RuntimeError("Cannot evaluate an empty dataloader")
    dist.reduce(total_loss, dst=0, op=dist.ReduceOp.AVG)
    epoch_loss = total_loss.item()
    epoch_loss /= seen_batches # average loss across all data

    metrics = {"top1": acc_top1.compute().item()}
    if acc_topk is not None:
        metrics[f"top{top_k}"] = acc_topk.compute().item()
    if return_metrics:
        return metrics, epoch_loss
    if top_k > 1:
        return metrics[f"top{top_k}"], epoch_loss
    return metrics["top1"], epoch_loss

def benchmark_model(
    cfg, model, iterations, trainloader, testloader, mixup=None, nomixup=False, verbose=0
    ):
    """
    Benchmarks model
        1. Train/Test dataloader throughput
        2. Model forward/backward throughput
        3. Pipeline (dataloader + model fwd/bwd) throughput

    Args
    Inputs:
        cfg: yacs config
        model: Pytorch model to benchmark
        iterations: number of iterations to benchmark
        trainloader: train dataloader
        testloader: test dataloader
        mixup: mixup for train dataloader (default: None)
        nomixup: If true, don't apply mixup during benchmark (on train dataloader)
        verbose (int): 0, 1: No intermediate prints, 2: print every iterations

    Outputs: (averaged per GPU)
        Train dataloader throughput (FPS/GPU)
        Test dataloader throughput
        Model forward/backword throughput
        Model forward throughput
        Train pipeline throughput
        Test pipeline throughput
    """
    # Read parameters from config
    batch_size = cfg.TRAIN.BATCHPERGPU
    dataset_name = cfg.TRAIN.DATASET
    device_rank = cfg.RANK
    use_amp = cfg.TRAIN.AMP

    # initialize tensors to store benchmark results. Will use torch.dist to average them across gpus
    trainloader_time = torch.zeros(1, dtype=torch.float32, device=device_rank)
    testloader_time = torch.zeros(1, dtype=torch.float32, device=device_rank)
    modelfbp_time = torch.zeros(1, dtype=torch.float32, device=device_rank)
    modelfwd_time = torch.zeros(1, dtype=torch.float32, device=device_rank)
    trainpl_time = torch.zeros(1, dtype=torch.float32, device=device_rank)
    testpl_time = torch.zeros(1, dtype=torch.float32, device=device_rank)

    imgs_tr = torch.zeros(1, dtype=torch.int64, device=device_rank)
    imgs_test = torch.zeros(1, dtype=torch.int64, device=device_rank)
    imgs_fbp = torch.zeros(1, dtype=torch.int64, device=device_rank)
    imgs_fwd = torch.zeros(1, dtype=torch.int64, device=device_rank)
    imgs_trp = torch.zeros(1, dtype=torch.int64, device=device_rank)
    imgs_tsp = torch.zeros(1, dtype=torch.int64, device=device_rank)

    # Benchmark dataloader
    trainloader_time[0], imgs_tr[0] = bm.benchmark_dataloader(trainloader, iterations, 
        device=device_rank, dtype=torch.float32, mixup=None if nomixup else mixup, verbose=(verbose>1) and device_rank==0)
    
    testloader_time[0], imgs_test[0] = bm.benchmark_dataloader(testloader, iterations, 
        device=device_rank, dtype=torch.float32, verbose=(verbose>1) and device_rank==0)

    if (verbose>0) and device_rank==0:
        logging.info("Dataloader benchmark done")
    
    modelfbp_time[0], imgs_fbp[0] = bm.benchmark_modelfbp_stage(
        model, cfg.MODEL.STAGE, batch_size, (batch_size,), max(iterations, 300), 'fbp',
        use_amp=use_amp, amp_dtype=utils.get_dtype(cfg.MODEL.AMPDTYPE),
        numeric_feature_dim=cfg.MODEL.NUMERIC_FEATURE_DIM, rlc_lmax=cfg.MODEL.RLC_LMAX,
        device=device_rank, verbose=(verbose>1) and device_rank==0)
    modelfwd_time[0], imgs_fwd[0] = bm.benchmark_modelfbp_stage(
        model, cfg.MODEL.STAGE, batch_size, (batch_size,), max(iterations, 300), 'fwd',
        use_amp=use_amp, amp_dtype=utils.get_dtype(cfg.MODEL.AMPDTYPE),
        numeric_feature_dim=cfg.MODEL.NUMERIC_FEATURE_DIM, rlc_lmax=cfg.MODEL.RLC_LMAX,
        device=device_rank, verbose=(verbose>1) and device_rank==0)

    if (verbose>0) and device_rank==0:
        logging.info("Model forward/backward benchmark done")

    # Benchmark pipeline
    trainpl_time[0], imgs_trp[0] = bm.benchmark_pipeline(
        model, trainloader, iterations, use_amp=use_amp, amp_dtype=utils.get_dtype(cfg.MODEL.AMPDTYPE),
        mode='train', device=device_rank, verbose=(verbose>1) and device_rank==0
    )
    testpl_time[0], imgs_tsp[0] = bm.benchmark_pipeline(
        model, testloader, iterations, use_amp=use_amp, amp_dtype=utils.get_dtype(cfg.MODEL.AMPDTYPE),
        mode='test', device=device_rank, verbose=(verbose>1) and device_rank==0
    )

    if (verbose>0) and device_rank==0:
        logging.info("Model forward/backward benchmark done")
    dist.barrier() # wait for all processes to complete benchmark

    # Reduce benchmark results across all gpu processes
    dist.reduce(trainloader_time, 0, op=dist.ReduceOp.SUM)
    dist.reduce(testloader_time, 0, op=dist.ReduceOp.SUM)
    dist.reduce(imgs_tr, 0, op=dist.ReduceOp.SUM)
    dist.reduce(imgs_test, 0, op=dist.ReduceOp.SUM)

    dist.reduce(modelfbp_time, 0, op=dist.ReduceOp.SUM)
    dist.reduce(modelfwd_time, 0, op=dist.ReduceOp.SUM)
    dist.reduce(imgs_fbp, 0, op=dist.ReduceOp.SUM)
    dist.reduce(imgs_fwd, 0, op=dist.ReduceOp.SUM)

    dist.reduce(trainpl_time, 0, op=dist.ReduceOp.SUM)
    dist.reduce(testpl_time, 0, op=dist.ReduceOp.SUM)
    dist.reduce(imgs_trp, 0, op=dist.ReduceOp.SUM)
    dist.reduce(imgs_tsp, 0, op=dist.ReduceOp.SUM)

    trainloader_fps = imgs_tr[0]/trainloader_time[0]
    testloader_fps = imgs_test[0]/testloader_time[0]
    modelfbp_fps = imgs_fbp[0]/modelfbp_time[0]
    modelfwd_fps = imgs_fwd[0]/modelfwd_time[0]
    trainpipe_fps = imgs_trp[0]/trainpl_time[0]
    testpipe_fps = imgs_tsp[0]/testpl_time[0]

    if (verbose>0) and device_rank==0:
        logging.info(f"Train loader: {trainloader_fps:.2f} FPS per GPU")
        logging.info(f"Model F/B pass: {modelfbp_fps:.2f} FPS per GPU")
        logging.info(f"Train pipeline: {trainpipe_fps:.2f} FPS per GPU")

        logging.info(f"Test loader: {testloader_fps:.2f} FPS per GPU")
        logging.info(f"Model Fwd pass: {modelfwd_fps:.2f} FPS per GPU")
        logging.info(f"Test pipeline: {testpipe_fps:.2f} FPS per GPU")
    
    return trainloader_fps, testloader_fps, modelfbp_fps, modelfwd_fps, trainpipe_fps, testpipe_fps
