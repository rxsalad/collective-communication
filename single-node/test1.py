'''
Environment: A single node with multiple GPUs, 
Image: rocm/vllm:latest 

This script demonstrates distributed GPU communication with PyTorch:
- Launches one process per GPU.
- Each process creates a tensor [rank * 10.0].
- Performs an all-reduce sum so that all GPUs end up with the same tensor [30.0].

0 + 10 + 20 = 30

- Prints the result on the master rank (rank 0).

Use case: This is a basic building block for distributed training with DDP, where gradients are summed across multiple GPUs.
'''

import torch
import torch.distributed as dist
import os

def run_ddp(rank, world_size): # Run by each process

    """
    Initializes the distributed environment and performs an all-reduce operation.
    """
    # The 'nccl' backend is automatically chosen when using ROCm.
    # the RCCL can automatically detect GPU topology (XGMI, PCIe, etc.)
    # XGMI - Cross GPU Micro Interconnect
    # All ranks join the communication group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # Set the device to the current rank's GPU
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Create a tensor (rank x 10.0) on the GPU
    tensor = torch.tensor([rank * 10.0], device=device)
    print(f"Rank {rank}: Initial tensor = {tensor}")
    # Perform an all-reduce operation - sum
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    #print(f"Rank {rank}: Tensor after all_reduce = {tensor}")

    if rank == 0:
        print(30 * "-" + " Results " + 30 * "-")   
        print(f"All-reduce result: {tensor}")   
    # Clean up the distributed environment
    dist.destroy_process_group()
 
if __name__ == "__main__":

    # Determine the number of available GPUs
    #world_size = torch.cuda.device_count() 
    world_size = 3 # Only use 3 GPUs for this example
    print(f"Running DDP on {world_size} GPUs.")
    
    # Launch multiple processes, one for each GPU
    #  each process gets a unique ID
    # Rank 0, serves as the master process ( save checkpoints, etc)
    # The main Python process (the one that called spawn) will wait until all spawned processes exit if join=True.
    torch.multiprocessing.spawn(run_ddp, args=(world_size,), nprocs=world_size, join=True)