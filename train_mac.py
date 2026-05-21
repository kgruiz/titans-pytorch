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
import tqdm
import gzip
import numpy as np
import importlib.metadata
from pathlib import Path

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
    parser.add_argument('--save-final-model', action = 'store_true', help = 'save the final model checkpoint when training finishes')
    parser.add_argument('--checkpoint-every', type = int, default = 0, help = 'save a checkpoint every N training steps; 0 disables periodic checkpoints')
    parser.add_argument('--checkpoint-dir', type = Path, default = Path('checkpoints'), help = 'directory for saved model checkpoints')
    parser.add_argument('--checkpoint-prefix', default = 'train_mac', help = 'filename prefix for saved model checkpoints')
    return parser.parse_args()

args = parse_args()

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

def log_runtime_config():
    print(f'using device: {DEVICE}')

    if USE_CUDA:
        major, minor = torch.cuda.get_device_capability()
        print(f'cuda device: {torch.cuda.get_device_name()}')
        print(f'cuda capability: sm_{major}{minor}')
        print(f'torch: {torch.__version__}')
        print(f'torch cuda: {torch.version.cuda}')
        try:
            triton_version = importlib.metadata.version('triton')
        except importlib.metadata.PackageNotFoundError:
            triton_version = 'not installed'

        print(f'triton: {triton_version}')
        print(f'triton support: {"available" if USE_TRITON else "unavailable"}')

    print(f'accelerated scan: {"enabled" if USE_ACCELERATED_SCAN else "disabled"}')
    print(f'flex attention: {"enabled" if USE_FLEX_ATTN else "disabled"}')
    print(f'fast inference cache: {"enabled" if USE_FAST_INFERENCE else "disabled"}')
    print(f'periodic checkpoints: {"enabled" if args.checkpoint_every > 0 else "disabled"}')
    print(f'final model save: {"enabled" if args.save_final_model else "disabled"}')

    if args.checkpoint_every > 0 or args.save_final_model:
        print(f'checkpoint dir: {args.checkpoint_dir}')

log_runtime_config()

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
            num_batches = NUM_BATCHES,
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
    print(f'saved checkpoint: {checkpoint_path}')

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10., desc = 'training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader), return_loss = True)
        loss.backward()

    print(f'training loss: {loss.item():.4f}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()
    wandb.log(dict(loss = loss.item()))

    step = i + 1

    if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
        save_checkpoint(step, loss = loss)

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader), return_loss = True)
            print(f'validation loss: {loss.item():.4f}')

    if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.sample(inp[None, ...], GENERATE_LENGTH, use_cache = USE_FAST_INFERENCE)
        output_str = decode_tokens(sample[0])
        print(output_str)

if args.save_final_model:
    save_checkpoint(NUM_BATCHES, final = True)
