'''
Environment: A single node with multiple GPUs, 
Image: rocm/vllm:latest 

This script demonstrates basic send/receive communication between two GPUs using PyTorch DDP/NCCL:
- Rank 0 sends a tensor to rank 1.
- Rank 1 receives the tensor from rank 0.

Only works with at least 2 GPUs.
Useful for understanding point-to-point communication in distributed GPU programs, unlike collective operations like all-reduce.

'''

import torch
import torch.distributed as dist
import os
def run_send_recv(rank, world_size):

    """
    Demonstrates point-to-point Send/Recv using RCCL/NCCL.
    """
    # Initialize distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # Set GPU for this rank
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Example: each rank creates a tensor
    tensor = torch.tensor([rank * 10.0], device=device)
    print(f"Rank {rank}: Initial tensor = {tensor}")

    if world_size < 2:
        raise ValueError("This example requires at least 2 ranks.")
    # Example: rank 0 sends tensor to rank 1
    if rank == 0:
        print(f"Rank {rank} sending tensor {tensor} to rank 1")
        dist.send(tensor=tensor, dst=1)
    elif rank == 1:
        # Creates a 1-element tensor filled with 0.0.
        recv_tensor = torch.zeros(1, device=device) 
        dist.recv(tensor=recv_tensor, src=0)
        print(f"Rank {rank} received tensor {recv_tensor} from rank 0")

    # Clean up
    dist.destroy_process_group()
 
if __name__ == "__main__":

    world_size = 2  # Number of GPUs/processes
    print(f"Running Send/Recv example on {world_size} GPUs.")
    torch.multiprocessing.spawn(run_send_recv, args=(world_size,), nprocs=world_size, join=True)
