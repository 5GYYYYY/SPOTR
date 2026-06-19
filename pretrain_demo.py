import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from SPOTR import SPOTREncoder, SPOTRDecoder


class DemoSignalDataset(Dataset):
    """
    A minimal synthetic signal dataset for SPOTR pre-training.

    Each sample is a physiological-signal-like tensor with shape [C, T].
    The DataLoader will batch it into [B, C, T], which is the input shape
    expected by SPOTREncoder.

    To use your own data, replace this class and return a float tensor
    with shape [C, T] in __getitem__.
    """

    def __init__(self, channels=12, signal_length=2000):
        self.num_samples = 100000
        self.channels = channels
        self.signal_length = signal_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        t = torch.linspace(0, 1, self.signal_length)
        signal = []

        for c in range(self.channels):
            freq = 1.0 + (index % 5)
            phase = 0.03 * (index % 16) + 0.05 * c
            wave = torch.sin(2 * torch.pi * freq * t + phase)
            wave = wave + 0.2 * torch.sin(2 * torch.pi * (freq * 2.0) * t + phase)
            wave = wave + 0.01 * torch.randn_like(wave)
            signal.append(wave)

        return torch.stack(signal, dim=0).float()


def main():
    parser = argparse.ArgumentParser("SPOTR pre-training demo")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--encoder_dim", type=int, default=512)
    parser.add_argument("--decoder_dim", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=100)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--encoder_layers", type=int, default=12)
    parser.add_argument("--decoder_layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(0)

    dataset = DemoSignalDataset(
        channels=args.channels,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    encoder = SPOTREncoder(
        encoder_dim=args.encoder_dim,
        patch_size=args.patch_size,
        num_heads=args.num_heads,
        n_layers=args.encoder_layers,
    ).to(args.device)
    decoder = SPOTRDecoder(
        encoder_dim=args.encoder_dim,
        decoder_dim=args.decoder_dim,
        patch_size=args.patch_size,
        num_heads=args.num_heads,
        n_layers=args.decoder_layers,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    encoder.train()
    decoder.train()

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")

        for x in pbar:
            # x: [B, C, T]
            x = x.to(args.device)

            x_embedding, x_target = encoder.encode(x)
            loss = decoder(x_embedding, x_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                args.grad_clip,
            )
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    torch.save(encoder.state_dict(), os.path.join(args.output_dir, "spotr_encoder_last.pt"))
    torch.save(decoder.state_dict(), os.path.join(args.output_dir, "spotr_decoder_last.pt"))


if __name__ == "__main__":
    main()
