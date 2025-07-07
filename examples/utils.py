import os
import torch
import pandas as pd
from datetime import timedelta
import torch.distributed as dist

from compressor import UHCCompressor
from torch import Future

try:
    from hadamard_cuda import hadamard_transform
except ImportError:
    hadamard_transform = None

INTEG_PARTITION_LAYER = 3
CHUNK_SIZE_THRESHOLD = 1 << 23 # 2^24 for language tasks and 2^23 for image tasks

_initialized = False
_random_diag_encode = None
_random_diag_decode = None

def thc_compress_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision floating-point format (``torch.float16``)
    and then divides it by the process group size.
    It allreduces those ``float16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    
    def compress_reduce_decompress(l, r, index, i, max_norm):
        compressed_vec = state["compressor"].compress(vec[l:r], (index, i), max_norm)
        reduced_fut = dist.all_reduce(compressed_vec, async_op=True, op=dist.ReduceOp.SUM).get_future()
        return reduced_fut.then(\
            lambda fut: ret_tensor[l:r].copy_(state["compressor"].decompress(fut.value()[0], (index, i), max_norm)))

    def return_func(fut):
        return ret_tensor
    
    state = state[0]
    if state["batch_idx"] == 0:
        return (
            dist.all_reduce(bucket.buffer(), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )
        
    else:
        index = bucket.index()
        vec = bucket.buffer()
        total_size = vec.numel()

        # Lazy init if this is the first time we see this bucket index
        if index not in state["partition_len"]:
            orig_total_size = total_size
            i = 0
            while True:
                if i >= INTEG_PARTITION_LAYER - 1 and total_size <= CHUNK_SIZE_THRESHOLD:
                    cur_size = total_size
                    cur_d = cur_size if (1 << (total_size.bit_length() - 1)) == cur_size else (1 << (total_size.bit_length()))
                else:
                    cur_size = min(1 << (total_size.bit_length() - 1), CHUNK_SIZE_THRESHOLD)
                    cur_d = cur_size

                state["params"]["d"][(index, i)] = cur_d
                state["params"]["size"][(index, i)] = cur_size
                state["start_idx"][(index, i)] = orig_total_size - total_size
                total_size -= cur_size
                i += 1
                if total_size == 0:
                    break

            state["partition_len"][index] = i
            state["ret_tensor"][index] = torch.zeros((orig_total_size), dtype=vec.dtype, device=vec.device)
            state["compressor"]: UHCCompressor = UHCCompressor(state["params"])

        ret_tensor = state["ret_tensor"][index]
        max_norms = []
        futures = []

        for i in range(state["partition_len"][index]):
            l = state["start_idx"][(index, i)]
            r = l + state["params"]["size"][(index, i)]
            self_norm = vec[l:r].norm(2).view(-1)
            max_norms.append(self_norm.item())
        max_norms = dist.all_reduce(torch.tensor(max_norms, device=state["params"]["device"]), async_op=True, op=dist.ReduceOp.MAX).get_future().wait()[0].tolist()

        for i in range(state["partition_len"][index]):
            l = state["start_idx"][(index, i)]
            r = l + state["params"]["size"][(index, i)]
            cur_future = compress_reduce_decompress(l, r, index, i, max_norms[i])
            futures.append(cur_future)

        return torch.futures.collect_all(futures).then(return_func)

def is_hadamard_available():
    return hadamard_transform is not None

def _initialize_hadamard_matrices():
    global _initialized, _random_diag_encode, _random_diag_decode
    if not _initialized:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sgen = torch.Generator(device=device)
        rgen = torch.Generator(device=device)
        sgen.manual_seed(0)
        rgen.manual_seed(0)
        _random_diag_encode = 2 * torch.bernoulli(torch.ones(250000000, device=device) / 2, generator=sgen) - 1
        _random_diag_decode = 2 * torch.bernoulli(torch.ones(250000000, device=device) / 2, generator=rgen) - 1
        _initialized = True

def hadamard_hook_cuda(process_group, bucket):
    # Initialize the matrices if not already done
    _initialize_hadamard_matrices()
    
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    tensor = bucket.buffer()
    tensor.div_(group_to_use.size())
    torch.cuda.synchronize()
    encoded_allreduced_tensor = dist.all_reduce(
        hadamard_transform(tensor * _random_diag_encode[:tensor.numel()]),
        group=group_to_use, 
        async_op=True
    )
    def decode(fut):
        decoded = hadamard_transform(fut.value()[0]) / len(fut.value()[0])
        return decoded * _random_diag_decode[:tensor.numel()]
    return encoded_allreduced_tensor.get_future().then(decode)

def setup_distributed_env(args):
    file_prefix = f"{args.comm}_{args.algo}_{args.model}_{args.epochs}_{args.batch_size}"
    os.environ['MASTER_PORT'] = '12355'
    os.environ['GLOO_SOCKET_IFNAME'] = args.dev
    os.environ['GLOO_ALGO'] = args.algo.capitalize()
    os.environ['GLOO_DPDK_TIMEOUT'] = str(args.tr_timeout)
    os.environ['GLOO_DPDK_THREADS_OFFSET'] = str(args.tr_threads_offset)
    os.environ['GLOO_DPDK_FILE_PREFIX'] = file_prefix
    return file_prefix + ".log"
    
def initialize_process_group(args):
    dist.init_process_group(
        backend=args.comm, 
        rank=int(args.nr), 
        world_size=int(args.nodes), 
        timeout=timedelta(seconds=200)
    )

def log_training_metrics(epoch_times, epoch_acc, epoch_loss, file_path):
    df1 = pd.DataFrame(list(zip(epoch_times, epoch_acc, epoch_loss)),
        columns=['Time', 'Train Acc', 'Train Loss'])
    df1.to_csv(file_path)

def calculate_classification_accuracy(predictions, labels):
    preds = torch.argmax(predictions, dim=1)
    return torch.mean((preds == labels).float())


def calculate_span_prediction_accuracy(outputs, start_positions, end_positions):
    start_preds = torch.argmax(outputs.start_logits, dim=1)
    end_preds = torch.argmax(outputs.end_logits, dim=1)
    
    accuracy = ((start_preds == start_positions).float() + 
               (end_preds == end_positions).float()) / 2.0
    
    return torch.mean(accuracy)
