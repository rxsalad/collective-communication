import torch
import torch.distributed as dist
import os

def main():
    rank = 1
    world_size = 2

    # Initialize distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set GPU for this rank
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Tensor to receive
    recv_tensor = torch.zeros(1, device=device)
    dist.recv(tensor=recv_tensor, src=0)
    print(f"Rank {rank} received tensor {recv_tensor} from rank 0")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
