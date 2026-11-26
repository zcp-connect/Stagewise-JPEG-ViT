import logging
import torch
import torch.optim as optim

from time import time
from torch import nn


def _move_mcu_batch(inputs, labels, device):
    if isinstance(inputs, dict):
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
    elif isinstance(inputs, (tuple, list)):
        inputs = tuple(item.to(device, non_blocking=True) for item in inputs)
    else:
        inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return inputs, labels


def make_synthetic_input(stage, batch_size, device, numeric_feature_dim=384, rlc_lmax=320):
    stage = stage.lower()
    if stage in ("rgb", "ycbcr"):
        return torch.randn(batch_size, 3, 224, 224, device=device)
    if stage in ("dct_float", "quant_dct", "dct"):
        return torch.randn(batch_size, 196, numeric_feature_dim, device=device)
    if stage in ("raw_byte", "mcu_bytes", "raw_bytes", "bitstream"):
        payload = torch.randint(0, 256, (batch_size, 196, rlc_lmax), device=device, dtype=torch.uint8)
        byte_mask = torch.ones((batch_size, 196, rlc_lmax), device=device, dtype=torch.bool)
        return payload, byte_mask
    if stage in ("rlc_dc", "rlc_dcac"):
        kind = torch.randint(0, 5, (batch_size, 196, rlc_lmax), device=device, dtype=torch.long)
        component = torch.randint(0, 4, (batch_size, 196, rlc_lmax), device=device, dtype=torch.long)
        block = torch.randint(0, 7, (batch_size, 196, rlc_lmax), device=device, dtype=torch.long)
        zigzag = torch.randint(0, 65, (batch_size, 196, rlc_lmax), device=device, dtype=torch.long)
        run = torch.randint(0, 17, (batch_size, 196, rlc_lmax), device=device, dtype=torch.long)
        size = torch.randint(0, 17, (batch_size, 196, rlc_lmax), device=device, dtype=torch.long)
        amplitude = torch.randn(batch_size, 196, rlc_lmax, device=device)
        mask = torch.ones((batch_size, 196, rlc_lmax), device=device, dtype=torch.bool)
        return kind, component, block, zigzag, run, size, amplitude, mask
    raise ValueError(f"Unsupported stage for benchmark: {stage}")


def benchmark_dataloader(dataloader, itr: int = 100, device='cpu', dtype=torch.float32, mixup=None, verbose=True):
    inittime = time()
    idx = 0
    img_count = 0
    init_took = 0
    time_took = 0

    while idx < itr:
        for i, data in enumerate(dataloader):
            if i == 0:
                starttime = time()
            if idx >= itr:
                break
            inputs, labels = data
            _move_mcu_batch(inputs, labels, device)
            img_count += labels.shape[0]
            if verbose:
                print(f"\rBenchmarking dataloader... {idx+1}/{itr}", end="", flush=True)
            idx += 1
        endtime = time()
        time_took += endtime - starttime
        init_took += starttime - inittime
        inittime = time()

    if verbose:
        print("\n", end="")
        logging.info(
            f"   Dataloader took: {time_took:.2f} sec for {idx} itrs / {img_count} imgs. "
            f"{img_count/time_took:.2f} FPS. Init took {init_took:.2f} sec"
        )
    return time_took, img_count


def benchmark_modelfbp_stage(
        model, stage="raw_byte", batch_size=128, outshape=(128,), itr: int = 100,
        mode='fbp', criterion=nn.CrossEntropyLoss(), use_amp=False,
        amp_dtype=torch.float16, numeric_feature_dim=384, rlc_lmax=384,
        device='cpu', verbose=True):
    inputs = make_synthetic_input(stage, batch_size, device, numeric_feature_dim=numeric_feature_dim, rlc_lmax=rlc_lmax)
    labels = torch.randint(0, 999, outshape, device=device, dtype=torch.int64)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0, eps=1e-4)
    gradscaler = torch.cuda.amp.GradScaler() if use_amp else None

    img_count = 0
    inittime = time()
    with torch.set_grad_enabled(mode == 'fbp'):
        for i in range(itr):
            if i == 0:
                starttime = time()
            if mode == 'fbp':
                optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                output = model(inputs)
                loss = criterion(output, labels)
                if mode == 'fbp':
                    if use_amp:
                        gradscaler.scale(loss).backward()
                        gradscaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                        gradscaler.step(optimizer)
                        gradscaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                        optimizer.step()
            img_count += outshape[0]
            if verbose:
                print(f"\rBenchmarking {stage} model {mode}... {i+1}/{itr}", end="", flush=True)

    endtime = time()
    time_took = endtime - starttime
    init_took = starttime - inittime
    if verbose:
        print("\n", end="")
        logging.info(
            f"   {stage} model {mode} (amp:{use_amp}) took: {time_took:.2f} sec for {itr} itrs / "
            f"{img_count} imgs. {img_count/time_took:.2f} FPS. Init took {init_took:.2f} sec"
        )
    return time_took, img_count


benchmark_modelfbp_mcu = benchmark_modelfbp_stage


def benchmark_pipeline(
        model, dataloader, itr, criterion=nn.CrossEntropyLoss(), use_amp=False,
        amp_dtype=torch.float16, mode='train', device='cpu', verbose=True):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0, eps=1e-4)
    gradscaler = torch.cuda.amp.GradScaler() if use_amp else None
    inittime = time()
    idx = 0
    init_took = 0
    time_took = 0
    img_count = 0

    with torch.set_grad_enabled(mode == 'train'):
        while idx < itr:
            for i, data in enumerate(dataloader):
                if i == 0:
                    starttime = time()
                if idx >= itr:
                    break
                inputs, labels = data
                inputs, labels = _move_mcu_batch(inputs, labels, device)
                if mode == 'train':
                    optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    output = model(inputs)
                    loss = criterion(output, labels)
                    if mode == 'train':
                        if use_amp:
                            gradscaler.scale(loss).backward()
                            gradscaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                            gradscaler.step(optimizer)
                            gradscaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                            optimizer.step()
                img_count += labels.shape[0]
                if verbose:
                    print(f"\rBenchmarking MCU pipeline... {idx+1}/{itr}", end="", flush=True)
                idx += 1
            endtime = time()
            time_took += endtime - starttime
            init_took += starttime - inittime
            inittime = time()

    if verbose:
        print("\n", end="")
        logging.info(
            f"   MCU pipeline (mode:{mode}, amp:{use_amp}) took: {time_took:.2f} sec for {idx} itrs / "
            f"{img_count} imgs. {img_count/time_took:.2f} FPS. Init took {init_took:.2f} sec"
        )
    return time_took, img_count
