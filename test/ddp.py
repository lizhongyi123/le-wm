
# ddp_example.py
import os
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


# ============================================================
# 1. 简单数据集
# ============================================================

class RandomDataset(Dataset):
    def __init__(self, n=10000, input_dim=32):
        self.x = torch.randn(n, input_dim)
        self.y = self.x.sum(dim=1, keepdim=True)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ============================================================
# 2. 简单模型
# ============================================================

class SimpleModel(nn.Module):
    def __init__(self, input_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. DDP 初始化和清理
# ============================================================

def setup_ddp():
    """
    torchrun 会自动设置这些环境变量：
        LOCAL_RANK: 当前进程在本机对应第几张 GPU
        RANK: 当前进程的全局编号
        WORLD_SIZE: 总进程数
    """
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return local_rank, rank, world_size, device


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


# ============================================================
# 4. 训练主函数
# ============================================================

def main():
    local_rank, rank, world_size, device = setup_ddp()

    if is_main_process():
        print("========== DDP Training ==========")
        print(f"world_size = {world_size}")

    print(f"rank={rank}, local_rank={local_rank}, device={device}")

    # --------------------------------------------------------
    # 数据集
    # --------------------------------------------------------
    dataset = RandomDataset(n=10000, input_dim=32)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # --------------------------------------------------------
    # 模型
    # --------------------------------------------------------
    model = SimpleModel(input_dim=32).to(device)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # --------------------------------------------------------
    # 训练
    # --------------------------------------------------------
    epochs = 10

    for epoch in range(epochs):
        sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if is_main_process():
            print(f"Epoch [{epoch + 1}/{epochs}], loss = {avg_loss:.6f}")

    # --------------------------------------------------------
    # 只让 rank=0 保存模型
    # --------------------------------------------------------
    if is_main_process():
        model_to_save = model.module
        torch.save(model_to_save.state_dict(), "ddp_example_model.pt")
        print("Saved model to ddp_example_model.pt")

    cleanup_ddp()


if __name__ == "__main__":
    main()