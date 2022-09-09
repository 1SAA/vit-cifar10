from distutils.command.config import config
import os
import psutil
import colossalai
import torch
import torch.distributed as dist
from functools import partial

from tqdm import tqdm
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.lr_scheduler import LinearWarmupLR
from colossalai.nn.optimizer import HybridAdam
from timm.models.vision_transformer import _create_vision_transformer, _cfg
from colossalai.gemini.update import ChunkManagerV2, search_chunk_configuration
from colossalai.gemini import GeminiManager
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils import get_current_device
from colossalai.nn.parallel import ZeroDDP
from colossalai.zero.zero_optimv2 import ZeroOptimizerV2
from data import build_data


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024 ** 2


def get_gpu_mem():
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def get_cur_gpu_mem():
    return torch.cuda.memory_allocated() / 1024 ** 2


def get_mem_info(prefix=''):
    return '{}current CUDA memory: {:.2f} MB, past max CUDA memory: {:.2f}, CPU memory {:.2f} MB'.format(
        prefix, get_cur_gpu_mem(), get_gpu_mem(), get_cpu_mem()
    )


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def vit_cifar(**kwargs):
    pretrained_cfg = _cfg(num_classes=10, input_size=(3, 32, 32), crop_pct=1.0)
    model_kwargs = dict(patch_size=4, embed_dim=512, depth=6, num_heads=8,
                        drop_rate=0.1, mlp_ratio=1.0, **kwargs)
    model = _create_vision_transformer('vit_cifar', pretrained_cfg=pretrained_cfg, **model_kwargs)
    return model


def train(
    epoch,
    sampler,
    model,
    loader,
    criterion,
    optimizer,
    show_progress=True,
    lr_scheduler=None):
    if sampler:
        sampler.set_epoch(epoch)
    model.train()
    train_iter = iter(loader)
    num_steps_per_epoch = len(loader)

    def run_step():
        optimizer.zero_grad()
        inputs, targets = next(train_iter)
        output = model(inputs)
        real_loss = criterion(output, targets)
        ret_loss = real_loss.item()
        optimizer.backward(real_loss)
        optimizer.step()
        return ret_loss

    with tqdm(range(num_steps_per_epoch), desc='train', ncols=0, disable=not show_progress) as t:
        for step in t:
            loss = run_step()
            # lr_scheduler.step()
            t.set_postfix(loss=f'{loss:.4f}')

    try:
        while True:
            next(train_iter)
    except StopIteration:
        pass


def evaluate(model, loader, show_progress=True):
    model.eval()
    total_number = 0
    correct_number = 0
    with torch.no_grad():
        with tqdm(loader, desc='valid', ncols=0, disable=not show_progress) as t:
            for inputs, targets in t:
                outputs = model(inputs)
                predict = torch.argmax(outputs, dim=-1)

                step_total = predict.size(0)
                step_correct = torch.sum(predict == targets).item()

                total_number += step_total
                correct_number += step_correct
    trans_buffer = torch.tensor([correct_number, total_number], device=get_current_device())
    dist.all_reduce(trans_buffer)
    return trans_buffer[0] / trans_buffer[1]


def train_cifar():
    disable_existing_loggers()
    args = colossalai.get_default_parser().parse_args()
    colossalai.launch_from_torch(config=args.config)  # initialize colossal environment
    logger = get_dist_logger()

    local_rank = dist.get_rank()

    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    train_dataloader, valid_dataloader, train_sampler, valid_sampler = build_data(gpc.config.BATCH_SIZE)
    with ColoInitContext(device=get_current_device()):
        model = vit_cifar()
    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f'Model numel: {numel}', ranks=[0])
    logger.info(get_mem_info(), ranks=[0])

    config_dict = search_chunk_configuration(
        model=model,
        search_range_mb=8,
        search_interval_byte=512,
        min_chunk_size_mb=4,
        filter_exlarge_params=True
    )

    chunk_manager = ChunkManagerV2(
        config_dict,
        init_device=get_current_device())

    gemini_manager = GeminiManager('cuda', chunk_manager)
    model = ZeroDDP(model, gemini_manager, pin_memory=False)

    logger.info(chunk_manager, ranks=[0])
    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])

    optimizer = HybridAdam(model.parameters(), lr=gpc.config.LEARNING_RATE)
    optimizer = ZeroOptimizerV2(optimizer, model, initial_scale=2 ** 5, gpu_margin_mem_ratio=0.0)

    criterion = torch.nn.CrossEntropyLoss()
    steps_per_epoch = len(train_dataloader)

    lr_scheduler = LinearWarmupLR(optimizer=optimizer,
                                  total_steps=gpc.config.NUM_EPOCHS,
                                  warmup_steps=gpc.config.WARMUP_EPOCHS)

    for epoch in range(gpc.config.NUM_EPOCHS):
        if local_rank == 0:
            print("Epoch: {}".format(epoch))
        dist.barrier()
        train(
            epoch=epoch,
            sampler=train_sampler,
            model=model,
            loader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            show_progress=local_rank == 0
        )
        lr_scheduler.step()
        correct_percentage = evaluate(
            model=model,
            loader=valid_dataloader,
            show_progress=local_rank == 0
        )
        logger.info("eval correct percentage: {:.4f}".format(correct_percentage), ranks=[0])


if __name__ == '__main__':
    train_cifar()
