import argparse
import sys
from pathlib import Path

# Add project root to path to allow importing mebench.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from mebench.models.blackbox_ripper.factory import create_blackbox_ripper_generator
from mebench.models.gan import DCGANDiscriminator


def _build_transform(dataset_name: str, img_size: int) -> transforms.Compose:
    dataset_name = str(dataset_name)

    t = [
        transforms.Resize((int(img_size), int(img_size))),
    ]

    # Our BlackboxRipper generators are RGB. For grayscale datasets, replicate to 3 channels.
    if dataset_name in {"FashionMNIST", "EMNIST", "MNIST"}:
        t.append(transforms.Grayscale(num_output_channels=3))

    t.extend(
        [
            transforms.ToTensor(),
            # Upstream generator training uses [-1, 1] scaling (Normalize(0.5, 0.5)).
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(t)


def _load_dataset(dataset_name: str, data_root: Path, img_size: int):
    transform = _build_transform(dataset_name, img_size)
    if dataset_name == "CIFAR10":
        return datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform)
    if dataset_name == "SVHN":
        return datasets.SVHN(root=str(data_root), split="train", download=True, transform=transform)
    if dataset_name == "FashionMNIST":
        return datasets.FashionMNIST(
            root=str(data_root), train=True, download=True, transform=transform
        )
    if dataset_name == "EMNIST":
        return datasets.EMNIST(
            root=str(data_root),
            split="balanced",
            train=True,
            download=True,
            transform=transform,
        )
    raise ValueError(
        f"Unsupported dataset '{dataset_name}'. Supported: CIFAR10, SVHN, FashionMNIST, EMNIST"
    )


def _hinge_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def _hinge_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    return -d_fake.mean()


def _bce_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    real_targets = torch.ones_like(d_real)
    fake_targets = torch.zeros_like(d_fake)
    return F.binary_cross_entropy_with_logits(d_real, real_targets) + F.binary_cross_entropy_with_logits(
        d_fake, fake_targets
    )


def _bce_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    real_targets = torch.ones_like(d_fake)
    return F.binary_cross_entropy_with_logits(d_fake, real_targets)


def train() -> None:
    parser = argparse.ArgumentParser(
        description="Train a pretrained generator checkpoint for Black-Box Ripper"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["CIFAR10", "SVHN", "FashionMNIST", "EMNIST"],
        help="Proxy dataset to train the generator on",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="cifar_sngan",
        choices=["cifar_sngan", "cifar_progan"],
        help="Generator architecture (upstream-faithful)",
    )
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--loss", choices=["hinge", "bce"], default="hinge")
    parser.add_argument("--disc-base-channels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for generator state_dict (.pth)",
    )

    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_root = project_root / "data"
    dataset = _load_dataset(args.dataset, data_root, args.img_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        drop_last=True,
    )

    gen = create_blackbox_ripper_generator(args.generator, device)
    latent_dim = int(getattr(gen, "latent_dim", 128))

    disc = DCGANDiscriminator(
        input_channels=3,
        base_channels=int(args.disc_base_channels),
        num_classes=None,
        input_size=int(args.img_size),
    ).to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=float(args.lr_g), betas=(float(args.beta1), float(args.beta2)))
    opt_d = torch.optim.Adam(disc.parameters(), lr=float(args.lr_d), betas=(float(args.beta1), float(args.beta2)))

    if args.loss == "hinge":
        d_loss_fn = _hinge_d_loss
        g_loss_fn = _hinge_g_loss
    else:
        d_loss_fn = _bce_d_loss
        g_loss_fn = _bce_g_loss

    out_path: Path
    if args.out is None:
        out_dir = project_root / "checkpoints" / "blackbox_ripper"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.dataset.lower()}_{args.generator}_seed{int(args.seed)}.pth"
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training generator={args.generator} on dataset={args.dataset}")
    print(f"Saving generator weights to: {out_path}")

    for epoch in range(int(args.epochs)):
        gen.train(True)
        disc.train(True)

        running_g = 0.0
        running_d = 0.0
        n_steps = 0

        data_iter = iter(loader)
        for step in range(int(args.steps_per_epoch)):
            try:
                real_x, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                real_x, _ = next(data_iter)

            real_x = real_x.to(device)

            # Discriminator step
            z = torch.randn(real_x.size(0), latent_dim, device=device)
            fake_x = gen(z)
            if fake_x.shape[-1] != real_x.shape[-1] or fake_x.shape[-2] != real_x.shape[-2]:
                fake_x = F.interpolate(
                    fake_x,
                    size=(int(args.img_size), int(args.img_size)),
                    mode="bilinear",
                    align_corners=False,
                )

            opt_d.zero_grad(set_to_none=True)
            d_real = disc(real_x).view(-1)
            d_fake = disc(fake_x.detach()).view(-1)
            loss_d = d_loss_fn(d_real, d_fake)
            loss_d.backward()
            opt_d.step()

            # Generator step
            z = torch.randn(real_x.size(0), latent_dim, device=device)
            fake_x = gen(z)
            if fake_x.shape[-1] != real_x.shape[-1] or fake_x.shape[-2] != real_x.shape[-2]:
                fake_x = F.interpolate(
                    fake_x,
                    size=(int(args.img_size), int(args.img_size)),
                    mode="bilinear",
                    align_corners=False,
                )
            opt_g.zero_grad(set_to_none=True)
            d_fake2 = disc(fake_x).view(-1)
            loss_g = g_loss_fn(d_fake2)
            loss_g.backward()
            opt_g.step()

            running_d += float(loss_d.item())
            running_g += float(loss_g.item())
            n_steps += 1

            if (step + 1) % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{int(args.epochs)} step {step+1}/{int(args.steps_per_epoch)} "
                    f"loss_d={running_d/n_steps:.4f} loss_g={running_g/n_steps:.4f}"
                )

        # Save checkpoint each epoch.
        gen.eval()
        torch.save(gen.state_dict(), out_path)

    print("Done.")


if __name__ == "__main__":
    train()
