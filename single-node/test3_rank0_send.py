'''
Environment: A single node with multiple GPUs, 
Image: rocm/vllm:latest 

This script demonstrates basic send/receive communication between two GPUs using PyTorch DDP/NCCL:
- Rank0_send.py, sends a tensor to rank 1.
- Rank1_recv.py receives the tensor from rank 0.

Only works with at least 2 GPUs.
Useful for understanding point-to-point communication in distributed GPU programs, unlike collective operations like all-reduce.
'''


import torch
import torch.distributed as dist
import os

def main():
    rank = 0
    world_size = 2

    # Initialize distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set GPU for this rank
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Tensor to send
    tensor = torch.tensor([rank * 10.0], device=device)
    print(f"Rank {rank}: Initial tensor = {tensor}")

    print(f"Rank {rank} sending tensor {tensor} to rank 1")
    dist.send(tensor=tensor, dst=1)

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
