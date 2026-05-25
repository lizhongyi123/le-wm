import torch
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

class PlainLeWMTrainer:
    """
    Plain PyTorch trainer for LeWM / JEPA world model.

    用来替代 Lightning Trainer + spt.Manager。
    """

    def __init__(
        self,
        model,
        sigreg,
        train_loader,
        val_loader,
        cfg,
        run_dir,
        output_model_name="lewm",
        device=None,
    ):
        self.model = model
        self.sigreg = sigreg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.output_model_name = output_model_name

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.sigreg.to(self.device)

        self.max_epochs = int(cfg.trainer.get("max_epochs", 100))
        self.grad_clip = float(cfg.trainer.get("gradient_clip_val", 0.0))

        self.optimizer = self.build_optimizer()

        self.best_val_loss = float("inf")
        self.global_step = 0
        self.start_epoch = 0

        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.last_ckpt_path = self.run_dir / f"{self.output_model_name}_last.ckpt"
        self.best_ckpt_path = self.run_dir / f"{self.output_model_name}_best.ckpt"

    def load_checkpoint(self, ckpt_path):
        ckpt_path = Path(ckpt_path)

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.sigreg.load_state_dict(ckpt["sigreg_state_dict"])

        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            # optimizer state 也要移动到当前 device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

        self.start_epoch = int(ckpt.get("epoch", 0))
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

        print(f"Loaded checkpoint: {ckpt_path}")
        print(f"Resume from epoch: {self.start_epoch}")
        print(f"Global step: {self.global_step}")
        print(f"Best val loss: {self.best_val_loss}")

    def build_optimizer(self):
        """
        根据 cfg.optimizer 创建 AdamW。
        如果你的 cfg.optimizer 里字段不同，可以在这里改。
        """
        lr = float(self.cfg.optimizer.get("lr", 1e-4))
        weight_decay = float(self.cfg.optimizer.get("weight_decay", 0.05))
        betas = tuple(self.cfg.optimizer.get("betas", [0.9, 0.95]))

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )

        return optimizer

    def move_to_device(self, batch):
        """
        把 batch 里的 tensor 移动到 GPU/CPU。
        """
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def compute_loss(self, batch):
        """
        对应原来的 lejepa_forward()。

        输入 batch:
            batch["pixels"]: (B, T, C, H, W)
            batch["action"]: (B, T, action_dim)

        输出:
            loss, pred_loss, sigreg_loss
        """
        ctx_len = self.cfg.wm.history_size
        n_preds = self.cfg.wm.num_preds
        lambd = self.cfg.loss.sigreg.weight

        # sequence 边界可能有 NaN，替换成 0
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        output = self.model.encode(batch)

        emb = output["emb"]          # (B, T, D)
        act_emb = output["act_emb"]  # (B, T, A_emb)

        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]

        tgt_emb = emb[:, n_preds:]
        pred_emb = self.model.predict(ctx_emb, ctx_act)

        if pred_emb.shape != tgt_emb.shape:
            raise RuntimeError(
                f"pred_emb 和 tgt_emb 形状不一致：\n"
                f"pred_emb.shape = {pred_emb.shape}\n"
                f"tgt_emb.shape  = {tgt_emb.shape}\n"
                f"通常需要 T = history_size + num_preds。\n"
                f"当前 emb.shape={emb.shape}, "
                f"history_size={ctx_len}, num_preds={n_preds}"
            )

        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        loss = pred_loss + lambd * sigreg_loss

        return {
            "loss": loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "emb": emb.detach(),
            "pred_emb": pred_emb.detach(),
            "tgt_emb": tgt_emb.detach(),
        }

    def train_one_epoch(self, epoch):
        self.model.train()
        self.sigreg.train()

        loss_sum = 0.0
        pred_sum = 0.0
        sigreg_sum = 0.0

        num_batches = len(self.train_loader)

        pbar = tqdm(
            enumerate(self.train_loader),
            total=num_batches,
            desc=f"Train Epoch {epoch + 1}/{self.max_epochs}",
            dynamic_ncols=True,
        )

        for step, batch in pbar:
            batch = self.move_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)

            output = self.compute_loss(batch)
            loss = output["loss"]

            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip,
                )

            self.optimizer.step()

            self.global_step += 1

            loss_value = output["loss"].item()
            pred_value = output["pred_loss"].item()
            sigreg_value = output["sigreg_loss"].item()

            loss_sum += loss_value
            pred_sum += pred_value
            sigreg_sum += sigreg_value

            pbar.set_postfix({
                "loss": f"{loss_value:.6f}",
                "pred": f"{pred_value:.6f}",
                "sigreg": f"{sigreg_value:.6f}",
            })

        return {
            "loss": loss_sum / max(1, num_batches),
            "pred_loss": pred_sum / max(1, num_batches),
            "sigreg_loss": sigreg_sum / max(1, num_batches),
        }

    @torch.no_grad()
    def validate_one_epoch(self, epoch):
        self.model.eval()
        self.sigreg.eval()

        loss_sum = 0.0
        pred_sum = 0.0
        sigreg_sum = 0.0

        num_batches = len(self.val_loader)

        for batch in self.val_loader:
            batch = self.move_to_device(batch)

            output = self.compute_loss(batch)

            loss_sum += output["loss"].item()
            pred_sum += output["pred_loss"].item()
            sigreg_sum += output["sigreg_loss"].item()

        return {
            "loss": loss_sum / max(1, num_batches),
            "pred_loss": pred_sum / max(1, num_batches),
            "sigreg_loss": sigreg_sum / max(1, num_batches),
        }

    def save_checkpoint(self, path, epoch, val_loss):
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "sigreg_state_dict": self.sigreg.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
        }

        torch.save(ckpt, path)

    def fit(self):
        print(f"Using device: {self.device}")
        print(f"Run directory: {self.run_dir}")
        print(f"Max epochs: {self.max_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")

        for epoch in range(self.start_epoch, self.max_epochs):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate_one_epoch(epoch)

            print(
                f"\nEpoch {epoch + 1}/{self.max_epochs}\n"
                f"  train/loss       = {train_metrics['loss']:.6f}\n"
                f"  train/pred_loss  = {train_metrics['pred_loss']:.6f}\n"
                f"  train/sigreg     = {train_metrics['sigreg_loss']:.6f}\n"
                f"  val/loss         = {val_metrics['loss']:.6f}\n"
                f"  val/pred_loss    = {val_metrics['pred_loss']:.6f}\n"
                f"  val/sigreg       = {val_metrics['sigreg_loss']:.6f}\n"
            )

            # 保存 last
            self.save_checkpoint(
                self.last_ckpt_path,
                epoch=epoch + 1,
                val_loss=val_metrics["loss"],
            )

            # 保存 best
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]

                self.save_checkpoint(
                    self.best_ckpt_path,
                    epoch=epoch + 1,
                    val_loss=val_metrics["loss"],
                )

                print(f"Saved best checkpoint: {self.best_ckpt_path}")

        print("Training finished.")
        print(f"Last checkpoint: {self.last_ckpt_path}")
        print(f"Best checkpoint: {self.best_ckpt_path}")