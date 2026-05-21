# /// script
# dependencies = [
#     "accelerate",
#     "adam-atan2-pytorch>=0.1.18",
#     "setuptools",
#     "titans-pytorch",
#     "tqdm",
#     "wandb"
# ]
# ///

import random
import argparse
import json
import subprocess
import tqdm
import gzip
import numpy as np
import importlib.metadata
from pathlib import Path
from datetime import datetime

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from adam_atan2_pytorch import AdoptAtan2

from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
    MemoryAttention
)

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = True
SEQ_LEN = 512

# neural memory related

NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 4
NEURAL_MEM_LAYERS = (2, 4, 6)                   # layers 2, 4, 6 have neural memory, can add more
NEURAL_MEM_GATE_ATTN_OUTPUT = False
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-1
USE_MEM_ATTENTION_MODEL = False
WINDOW_SIZE = 32
NEURAL_MEM_SEGMENT_LEN = 4                      # set smaller for more granularity for learning rate / momentum etc
NEURAL_MEM_BATCH_SIZE = 128                     # set smaller to update the neural memory weights more often as it traverses the sequence
SLIDING_WINDOWS = True
STORE_ATTN_POOL_CHUNKS = True                   # whether to use attention pooling for chunk derived momentum, per-layer lr mod, decay
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
NEURAL_MEM_WEIGHT_RESIDUAL = True               # learning to accept contributions from the weights of the previous neural mem layer brings about significant improvements. this was improvised and not in the paper, but inspired by the value residual learning free lunch paper
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True        # will allow the neural memory to select what layers from which to derive queries / keys / values, effectively allowing it to graft itself to the transformer in any way to be beneficial. this is to address an issue from a phd student who noted that the mem network is learning nothing more than wk @ wv. this also generalizes all possible ways to connect the neural memory to a transformer, a sort of NAS
NEURAL_MEM_SPEC_NORM_SURPRISES = True           # applying lessons from Muon optimizer to surprise updates, by spectral norming the surprises
NEURAL_MEM_STORE_WITH_LOOKAHEAD_VALUE = False   # store with values from next timestep - Sakana AI finding

# experiment related

PROJECT_NAME = 'titans-mac-transformer'
RUN_NAME = f'mac - {NUM_LONGTERM_MEM} longterm mems, layers {NEURAL_MEM_LAYERS}'
WANDB_ONLINE = False # turn this on to pipe experiment to cloud

# perf related

# accelerated-scan's Triton backward kernel can hit illegal memory access on the A10 setup.
USE_ACCELERATED_SCAN = False
USE_FLEX_ATTN = True
USE_FAST_INFERENCE = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-batches', type = int, default = NUM_BATCHES, help = 'number of optimizer-update training steps to run')
    parser.add_argument('--save-final-model', action = 'store_true', help = 'save the final model checkpoint when training finishes')
    parser.add_argument('--checkpoint-every', type = int, default = 0, help = 'save a checkpoint every N training steps; 0 disables periodic checkpoints')
    parser.add_argument('--checkpoint-dir', type = Path, default = Path('checkpoints'), help = 'directory for saved model checkpoints')
    parser.add_argument('--checkpoint-prefix', default = 'train_mac', help = 'filename prefix for saved model checkpoints')
    return parser.parse_args()

args = parse_args()
SAVE_CHECKPOINTS = args.checkpoint_every > 0 or args.save_final_model

if SAVE_CHECKPOINTS and args.checkpoint_dir == Path('checkpoints'):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    args.checkpoint_dir = args.checkpoint_dir / f'{args.checkpoint_prefix}-{timestamp}'

def cuda_is_supported():
    # CUDA can be visible even when the installed PyTorch build cannot run kernels for this GPU.
    if not torch.cuda.is_available():
        return False

    try:
        x = torch.ones(1, device = 'cuda')
        y = x + 1
        torch.cuda.synchronize()
        return y.item() == 2
    except Exception as err:
        print(f'CUDA is visible but unusable, falling back to CPU: {err}')
        return False

def triton_is_supported():
    # Flex attention and accelerated scan rely on Triton, which requires newer CUDA devices.
    if not torch.cuda.is_available():
        return False

    major, _ = torch.cuda.get_device_capability()
    return major >= 7

# Keep ordinary CUDA execution when it works, but fall back from unsupported CUDA / Triton paths.
DEVICE = torch.device('cuda' if cuda_is_supported() else 'cpu')
USE_CUDA = DEVICE.type == 'cuda'
USE_TRITON = USE_CUDA and triton_is_supported()

USE_ACCELERATED_SCAN = USE_ACCELERATED_SCAN and USE_TRITON
USE_FLEX_ATTN = USE_FLEX_ATTN and USE_TRITON
USE_FAST_INFERENCE = USE_FAST_INFERENCE and USE_CUDA

def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not installed'

def git_output(*args):
    try:
        return subprocess.check_output(('git', *args), text = True).strip()
    except Exception:
        return None

def runtime_info():
    info = dict(
        device = str(DEVICE),
        torch = torch.__version__,
        torch_cuda = torch.version.cuda,
        triton = package_version('triton'),
        triton_support = USE_TRITON,
        accelerated_scan = USE_ACCELERATED_SCAN,
        flex_attention = USE_FLEX_ATTN,
        fast_inference_cache = USE_FAST_INFERENCE,
    )

    if USE_CUDA:
        major, minor = torch.cuda.get_device_capability()
        info.update(
            cuda_device = torch.cuda.get_device_name(),
            cuda_capability = f'sm_{major}{minor}',
        )

    return info

def log_runtime_config():
    print(f'using device: {DEVICE}')

    if USE_CUDA:
        major, minor = torch.cuda.get_device_capability()
        print(f'cuda device: {torch.cuda.get_device_name()}')
        print(f'cuda capability: sm_{major}{minor}')
        print(f'torch: {torch.__version__}')
        print(f'torch cuda: {torch.version.cuda}')
        print(f'triton: {package_version("triton")}')
        print(f'triton support: {"available" if USE_TRITON else "unavailable"}')

    print(f'accelerated scan: {"enabled" if USE_ACCELERATED_SCAN else "disabled"}')
    print(f'flex attention: {"enabled" if USE_FLEX_ATTN else "disabled"}')
    print(f'fast inference cache: {"enabled" if USE_FAST_INFERENCE else "disabled"}')
    print(f'periodic checkpoints: {"enabled" if args.checkpoint_every > 0 else "disabled"}')
    print(f'final model save: {"enabled" if args.save_final_model else "disabled"}')

    if SAVE_CHECKPOINTS:
        print(f'checkpoint dir: {args.checkpoint_dir}')

log_runtime_config()

def write_run_info():
    if not SAVE_CHECKPOINTS:
        return

    args.checkpoint_dir.mkdir(parents = True, exist_ok = True)

    run_info = dict(
        created_at = datetime.now().isoformat(timespec = 'seconds'),
        checkpoint_dir = str(args.checkpoint_dir),
        cli_args = dict(
            num_batches = args.num_batches,
            save_final_model = args.save_final_model,
            checkpoint_every = args.checkpoint_every,
            checkpoint_dir = str(args.checkpoint_dir),
            checkpoint_prefix = args.checkpoint_prefix,
        ),
        training = dict(
            num_batches = args.num_batches,
            batch_size = BATCH_SIZE,
            gradient_accumulate_every = GRADIENT_ACCUMULATE_EVERY,
            learning_rate = LEARNING_RATE,
            validate_every = VALIDATE_EVERY,
            generate_every = GENERATE_EVERY,
            prime_length = PRIME_LENGTH,
            generate_length = GENERATE_LENGTH,
            should_generate = SHOULD_GENERATE,
            seq_len = SEQ_LEN,
        ),
        model = dict(
            neural_memory_depth = NEURAL_MEMORY_DEPTH,
            num_persist_mem = NUM_PERSIST_MEM,
            num_longterm_mem = NUM_LONGTERM_MEM,
            neural_mem_layers = NEURAL_MEM_LAYERS,
            neural_mem_gate_attn_output = NEURAL_MEM_GATE_ATTN_OUTPUT,
            neural_mem_momentum = NEURAL_MEM_MOMENTUM,
            neural_mem_momentum_order = NEURAL_MEM_MOMENTUM_ORDER,
            neural_mem_qk_norm = NEURAL_MEM_QK_NORM,
            neural_mem_max_lr = NEURAL_MEM_MAX_LR,
            use_mem_attention_model = USE_MEM_ATTENTION_MODEL,
            window_size = WINDOW_SIZE,
            neural_mem_segment_len = NEURAL_MEM_SEGMENT_LEN,
            neural_mem_batch_size = NEURAL_MEM_BATCH_SIZE,
            sliding_windows = SLIDING_WINDOWS,
            store_attn_pool_chunks = STORE_ATTN_POOL_CHUNKS,
            memory_model_per_layer_learned_lr = MEMORY_MODEL_PER_LAYER_LEARNED_LR,
            neural_mem_weight_residual = NEURAL_MEM_WEIGHT_RESIDUAL,
            neural_mem_qkv_receives_diff_view = NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
            neural_mem_spec_norm_surprises = NEURAL_MEM_SPEC_NORM_SURPRISES,
            neural_mem_store_with_lookahead_value = NEURAL_MEM_STORE_WITH_LOOKAHEAD_VALUE,
        ),
        runtime = runtime_info(),
        git = dict(
            commit = git_output('rev-parse', 'HEAD'),
            status = git_output('status', '--short'),
        ),
    )

    run_info_path = args.checkpoint_dir / 'run-info.json'
    run_info_path.write_text(json.dumps(run_info, indent = 2) + '\n')
    tqdm.tqdm.write(f'wrote run info: {run_info_path}')

write_run_info()

def log_metric(event):
    if not SAVE_CHECKPOINTS:
        return

    metrics_path = args.checkpoint_dir / 'metrics.jsonl'

    with metrics_path.open('a') as file:
        file.write(json.dumps(event) + '\n')

# wandb experiment tracker

import wandb
wandb.init(project = PROJECT_NAME, mode = 'disabled' if not WANDB_ONLINE else 'online')
wandb.run.name = RUN_NAME
wandb.run.save()

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# memory model

if USE_MEM_ATTENTION_MODEL:
    neural_memory_model = MemoryAttention(
        dim = 64
    )
else:
    neural_memory_model = MemoryMLP(
        dim = 64,
        depth = NEURAL_MEMORY_DEPTH
    )

# instantiate memory-as-context transformer

model = MemoryAsContextTransformer(
    num_tokens = 256,
    dim = 384,
    depth = 8,
    segment_len = WINDOW_SIZE,
    num_persist_mem_tokens = NUM_PERSIST_MEM,
    num_longterm_mem_tokens = NUM_LONGTERM_MEM,
    neural_memory_layers = NEURAL_MEM_LAYERS,
    neural_memory_segment_len = NEURAL_MEM_SEGMENT_LEN,
    neural_memory_batch_size = NEURAL_MEM_BATCH_SIZE,
    neural_mem_gate_attn_output = NEURAL_MEM_GATE_ATTN_OUTPUT,
    neural_mem_weight_residual = NEURAL_MEM_WEIGHT_RESIDUAL,
    neural_memory_qkv_receives_diff_views = NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
    use_flex_attn = USE_FLEX_ATTN,
    sliding_window_attn = SLIDING_WINDOWS,
    neural_memory_model = neural_memory_model,
    neural_memory_kwargs = dict(
        dim_head = 64,
        heads = 4,
        attn_pool_chunks = STORE_ATTN_POOL_CHUNKS,
        qk_rmsnorm = NEURAL_MEM_QK_NORM,
        momentum = NEURAL_MEM_MOMENTUM,
        momentum_order = NEURAL_MEM_MOMENTUM_ORDER,
        default_step_transform_max_lr = NEURAL_MEM_MAX_LR,
        use_accelerated_scan = USE_ACCELERATED_SCAN,
        per_parameter_lr_modulation = MEMORY_MODEL_PER_LAYER_LEARNED_LR,
        spectral_norm_surprises = NEURAL_MEM_SPEC_NORM_SURPRISES,
        store_with_lookahead_value = NEURAL_MEM_STORE_WITH_LOOKAHEAD_VALUE
    )
).to(DEVICE)

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
    data_train, data_val = np.split(data, [int(90e6)])
    data_train, data_val = map(torch.from_numpy, (data_train, data_val))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.to(DEVICE)

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# optimizer

optim = AdoptAtan2(model.parameters(), lr = LEARNING_RATE)

def save_checkpoint(step, loss = None, final = False):
    args.checkpoint_dir.mkdir(parents = True, exist_ok = True)

    suffix = 'final' if final else f'step-{step:06d}'
    checkpoint_path = args.checkpoint_dir / f'{args.checkpoint_prefix}-{suffix}.pt'
    checkpoint = dict(
        step = step,
        loss = None if loss is None else loss.item(),
        model = model.state_dict(),
        optim = optim.state_dict(),
        config = dict(
            num_batches = args.num_batches,
            batch_size = BATCH_SIZE,
            gradient_accumulate_every = GRADIENT_ACCUMULATE_EVERY,
            learning_rate = LEARNING_RATE,
            seq_len = SEQ_LEN,
            use_accelerated_scan = USE_ACCELERATED_SCAN,
            use_flex_attn = USE_FLEX_ATTN,
            use_fast_inference = USE_FAST_INFERENCE,
        )
    )

    torch.save(checkpoint, checkpoint_path)
    tqdm.tqdm.write(f'saved checkpoint: {checkpoint_path}')
    log_metric(dict(
        step = step,
        event = 'checkpoint',
        path = str(checkpoint_path),
        final = final,
        loss = None if loss is None else loss.item(),
    ))

# training

for i in tqdm.tqdm(range(args.num_batches), mininterval = 10., desc = 'training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader), return_loss = True)
        loss.backward()

    step = i + 1
    train_loss = loss.item()
    tqdm.tqdm.write(f'step {step} train loss: {train_loss:.4f}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()
    wandb.log(dict(loss = train_loss))
    log_metric(dict(step = step, event = 'train', loss = train_loss))

    if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
        save_checkpoint(step, loss = loss)

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader), return_loss = True)
            validation_loss = loss.item()
            tqdm.tqdm.write(f'step {step} validation loss: {validation_loss:.4f}')
            log_metric(dict(step = step, event = 'validation', loss = validation_loss))

    if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        tqdm.tqdm.write(f'--- sample step {step} prompt ---')
        tqdm.tqdm.write(prime)

        sample = model.sample(inp[None, ...], GENERATE_LENGTH, use_cache = USE_FAST_INFERENCE)
        output_str = decode_tokens(sample[0])
        tqdm.tqdm.write(f'--- sample step {step} output ---')
        tqdm.tqdm.write(output_str)
        tqdm.tqdm.write(f'--- end sample step {step} ---')
        log_metric(dict(step = step, event = 'sample', prompt = prime, output = output_str))

if args.save_final_model:
    save_checkpoint(args.num_batches, final = True)
