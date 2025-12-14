from .InfiniLLM import Llama

import torch
import pytorch_lightning as pl 
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import os


class LightLLama(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, lr=2e-5, pad_token_id=None, warmup_steps=500, total_steps=10000):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def training_step(self, batch, batch_idx):
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = self.model(input_ids, start_pos=0)  # [B, seq, vocab]

        logits = logits[:, :-1, :]
        targets = targets[:, 1:]

        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )

        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return max(
                0.0,
                float(self.total_steps - step) / float(max(1, self.total_steps - self.warmup_steps)),
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def collate_fn(batch, tokenizer, model_seq_len):
    texts = [item["text"] for item in batch]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=model_seq_len,
    )
    input_ids = enc["input_ids"]
    labels = input_ids.clone()

    return {"input_ids": input_ids.long(), "labels": labels.long()}


def main():
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en")["train"]

    chkpt_dir = os.environ["LLAMA_DIR"]
    model_seq_len = 512
    model_batch_size = 1
    tokenizer = AutoTokenizer.from_pretrained(chkpt_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = DataLoader(
        dataset,
        batch_size=model_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, model_seq_len),
        num_workers=2,
    )

    llama = Llama.LLama.build(chkpt_dir, model_seq_len, model_batch_size, False)

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=3,
        precision="bf16-true",
        log_every_n_steps=10,
        accumulate_grad_batches=8,  # simulate larger batch size
        gradient_clip_val=1.0
    )

    model = LightLLama(
        llama,
        lr=5e-4,
        pad_token_id=tokenizer.pad_token_id,
        warmup_steps=500,
        total_steps=len(dataloader) * trainer.max_epochs,
    )

    trainer.fit(model, train_dataloaders=dataloader)
    trainer.save_checkpoint("checkpoints/finetuned.ckpt", weights_only=True)


if __name__ == "__main__":
    main()
