import torch
import torch.distributed as dist
import os

def main():
    rank = 1
    world_size = 2

    # Node1 is the master
    os.environ['MASTER_ADDR'] = '10.108.2.223'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_IB_HCA'] = 'mlx5_0'  
    os.environ['NCCL_IB_GID_INDEX'] = '1'  
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set GPU for this rank
    #torch.cuda.set_device(rank)
    torch.cuda.set_device(0) 
    #device = torch.device(f"cuda:{rank}")
    device = torch.device(f"cuda:0")

    recv_tensor = torch.zeros(1, device=device) # 1 means 1-dimensional tensor with 1 element
    dist.recv(tensor=recv_tensor, src=0)        # src=0 means receive from rank 0 (sending process), global concept in the distributed group 
    print(f"Rank {rank} received tensor {recv_tensor} from rank 0")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
