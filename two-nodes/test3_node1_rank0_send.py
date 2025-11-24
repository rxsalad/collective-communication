'''

| Node  | IP Address   | Script                      | Rank |
| ----- | ------------ | --------------------------- | ---- |
| Node1 | 10.108.2.223 | `test3_node1_rank0_send.py` | 0    |
| Node2 | 10.108.2.74  | `test3_node2_rank1_recv.py` | 1    |

Rank identifies the process in the distributed group.
world_size = total number of processes in the distributed group = 2 here.

Node1 is the master node, responsible for coordinating the distributed group.
- Node1 starts and initializes the process group.
- Node2 starts and initializes the process group.
- Node1 creates tensor [0.0] and sends it to Node2.
- Node2 receives tensor [0.0] and prints it.
- Both nodes destroy the process group.

We can decouple which GPU is used from which fabric interface is used. 
They are independent resources, and PyTorch + NCCL allows you to map them flexibly.


The eth0 is used only to exchange NCCL unique IDs, ranks, and other handshake information. 
After bootstrap, all heavy GPU communication goes over RoCE (mlx5_0:1).



IB/RoCE setup is correct: mlx5_0 with GID_INDEX=1 is being used.
Communicator initialized successfully; P2P send/recv worked fine.

ROCm/RCCL environment is consistent, but you might consider:
- Disabling NUMA auto balancing for consistent performance.
- Adding iommu=pt to kernel boot parameters for stability on AMD GPUs.

GDRDMA (GPU Direct RDMA) is being used, which is optimal for P2P performance.
'''



import torch
import torch.distributed as dist
import os

def main():
    rank = 0
    world_size = 2

    # Node1 is the master
    os.environ['MASTER_ADDR'] = '10.108.2.223' # The IP address of Master Node
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_IB_HCA'] = 'mlx5_0'      # The InfiniBand/RoCE network interface to use for NCCL (high-speed GPU communication).
    os.environ['NCCL_IB_GID_INDEX'] = '1'      # GID index for RoCE v2.
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # The network interface for TCP fallback if IB isn’t available.

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set GPU for this rank
    #torch.cuda.set_device(rank)
    torch.cuda.set_device(0) 
    #device = torch.device(f"cuda:{rank}")
    device = torch.device(f"cuda:0")

    # Tensor to send
    tensor = torch.tensor([rank * 10.0], device=device)
    print(f"Rank {rank}: Initial tensor = {tensor}")

    print(f"Rank {rank} sending tensor {tensor} to rank 1")
    dist.send(tensor=tensor, dst=1)  # dst=1 means send to rank 1 (receiving process), global concept in the distributed group 

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
