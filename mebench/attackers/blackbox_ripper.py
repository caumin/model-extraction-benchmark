import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from mebench.attackers.base import BaseAttack
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.models.gan import (
    DCGANGenerator,
    DCGANDiscriminator,
    SNDCGANGenerator,
    SNDCGANDiscriminator,
)
from mebench.models.substitute_factory import create_substitute
from mebench.data.loaders import create_dataloader


class BlackboxRipper(BaseAttack):
    """Black-box Ripper attack with proxy GAN and evolutionary search."""

    def __init__(self, config: dict, state: BenchmarkState) -> None:
        super().__init__(config, state)

        self.batch_size = int(config.get("batch_size", 128))

        # Defaults aligned to the official Black-box Ripper implementation.
        # Repo: https://github.com/antoniobarbalau/black-box-ripper
        # - z ~ U(-3.3, 3.3)
        # - population size = 30, elite size = 10, mutation sigma = 0.5
        self.noise_dim = int(config.get("noise_dim", 128))
        self.latent_bound = float(config.get("latent_bound", 3.3))
        self.population_size = int(config.get("population_size", 30))
        self.elite_size = int(config.get("elite_size", 10))
        self.mutation_scale = float(config.get("mutation_scale", 0.5))
        self.fitness_threshold = float(config.get("fitness_threshold", 0.02))
        self.max_evolve_iters = int(config.get("max_evolve_iters", 10))
        self.pretrain_steps = int(config.get("pretrain_steps", 100))
        self.train_every = int(config.get("train_every", 500))
        self.substitute_lr = float(config.get("substitute_lr", 0.01))
        self.substitute_epochs = int(config.get("substitute_epochs", 5))
        self.base_channels = int(config.get("base_channels", 64))
        self.gan_backbone = str(config.get("gan_backbone", "sngan")).lower()
        self.num_classes = int(
            state.metadata.get("num_classes")
            or config.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )

        self.generator: nn.Module | None = None
        self.discriminator: nn.Module | None = None
        self.gen_optimizer: optim.Optimizer | None = None
        self.dis_optimizer: optim.Optimizer | None = None
        self.substitute: nn.Module | None = None
        self.substitute_optimizer: optim.Optimizer | None = None

        self.proxy_loader = None
        self.proxy_iter = None
        self.pretrained = False

        self._initialize_state(state)

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state["population"] = None
        state.attack_state["targets"] = None
        state.attack_state["evolve_iter"] = 0
        state.attack_state["reset_population"] = False
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []
        state.attack_state["substitute"] = None

    def _init_models(self, state: BenchmarkState) -> None:
        device = state.metadata.get("device", "cpu")

        if self.generator is None:
            gen_cls = DCGANGenerator
            if self.gan_backbone in {"sngan", "sndcgan", "sn-dcgan"}:
                gen_cls = SNDCGANGenerator

            self.generator = gen_cls(
                noise_dim=self.noise_dim,
                output_channels=int(self.config.get("output_channels", state.metadata.get("input_shape", (3, 32, 32))[0])),
                base_channels=self.base_channels,
                num_classes=None,
                output_size=int(state.metadata.get("input_shape", (3, 32, 32))[1]),
            ).to(device)
            self.gen_optimizer = optim.Adam(
                self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999)
            )

        if self.discriminator is None:
            dis_cls = DCGANDiscriminator
            if self.gan_backbone in {"sngan", "sndcgan", "sn-dcgan"}:
                dis_cls = SNDCGANDiscriminator

            self.discriminator = dis_cls(
                input_channels=int(self.config.get("output_channels", state.metadata.get("input_shape", (3, 32, 32))[0])),
                base_channels=self.base_channels,
                num_classes=None,
                input_size=int(state.metadata.get("input_shape", (3, 32, 32))[1]),
            ).to(device)
            self.dis_optimizer = optim.Adam(
                self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)
            )

        if self.proxy_loader is None:
            proxy_config = self.config.get("attack", {}).get("proxy_dataset")
            if proxy_config is None:
                proxy_config = self.config.get("proxy_dataset")
            if proxy_config is None:
                raise ValueError("BlackboxRipper requires proxy_dataset configuration")
            self.proxy_loader = create_dataloader(
                proxy_config,
                batch_size=self.batch_size,
                shuffle=True,
            )
            self.proxy_iter = iter(self.proxy_loader)

        if not self.pretrained and self.pretrain_steps > 0:
            self._pretrain_gan(device)
            self.pretrained = True

    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        self._init_models(state)
        device = state.metadata.get("device", "cpu")

        population_size = max(k, self.population_size)
        population = state.attack_state.get("population")
        targets = state.attack_state.get("targets")

        if population is not None:
            population = population.to(device)
        if targets is not None:
            targets = targets.to(device)

        evolve_iter = int(state.attack_state.get("evolve_iter", 0))
        reset_population = bool(state.attack_state.get("reset_population", False))

        if (
            population is None
            or targets is None
            or population.size(0) != population_size
            or reset_population
            or evolve_iter >= self.max_evolve_iters
        ):
            # Paper: initialize population from uniform U(-u, u)
            u = float(self.latent_bound)
            population = (torch.rand(population_size, self.noise_dim, device=device) * 2.0 - 1.0) * u
            targets = torch.randint(0, self.num_classes, (population_size,), device=device)
            state.attack_state["population"] = population.detach().cpu()
            state.attack_state["targets"] = targets.detach().cpu()
            state.attack_state["evolve_iter"] = 0
            state.attack_state["reset_population"] = False

        with torch.no_grad():
            x = self.generator(population)[:k]
        meta = {
            "population": population.detach().cpu(),
            "targets": targets.detach().cpu(),
            "population_size": population_size,
        }
        return QueryBatch(x=x, meta=meta)

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if self.generator is None:
            return

        state.attack_state["query_data_x"].append(query_batch.x.detach().cpu())
        state.attack_state["query_data_y"].append(oracle_output.y.detach().cpu())

        population = query_batch.meta.get("population")
        targets = query_batch.meta.get("targets")
        if population is not None and targets is not None:
            self._evolve_population(
                population.to(query_batch.x.device),
                targets.to(query_batch.x.device),
                oracle_output,
                state,
            )

        if state.query_count % self.train_every == 0 and state.query_count > 0:
            self._train_substitute(state)

    def _evolve_population(
        self,
        population: torch.Tensor,
        targets: torch.Tensor,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if oracle_output.kind == "soft_prob":
            victim_probs = oracle_output.y.to(population.device)
        else:
            victim_probs = F.one_hot(
                oracle_output.y, num_classes=self.num_classes
            ).float().to(population.device)

        # Fitness: minimize MSE to the one-hot target.
        targets = targets[: victim_probs.size(0)].to(population.device)
        target_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        diff = victim_probs - target_onehot
        mse = (diff * diff).mean(dim=1)
        fitness = -mse

        # Stop condition: if we found a sufficiently good sample, restart search.
        if mse.numel() > 0 and float(mse.min().item()) <= float(self.fitness_threshold):
            state.attack_state["reset_population"] = True

        elite_count = max(1, min(int(self.elite_size), population.size(0)))
        elite_idx = torch.topk(fitness, elite_count).indices
        elite = population[elite_idx]

        # Two mutation passes from elites (mirrors reference code pattern).
        mutated_1 = elite + torch.randn_like(elite) * self.mutation_scale
        mutated_2 = elite + torch.randn_like(elite) * self.mutation_scale
        new_population = torch.cat([elite, mutated_1, mutated_2], dim=0)
        if new_population.size(0) < population.size(0):
            pad = population.size(0) - new_population.size(0)
            u = float(self.latent_bound)
            pad_z = (torch.rand(pad, self.noise_dim, device=population.device) * 2.0 - 1.0) * u
            new_population = torch.cat([new_population, pad_z], dim=0)
        new_population = new_population[: population.size(0)]
        if self.latent_bound > 0:
            new_population = new_population.clamp(-self.latent_bound, self.latent_bound)

        state.attack_state["population"] = new_population.detach().cpu()
        state.attack_state["targets"] = targets.detach().cpu()
        state.attack_state["evolve_iter"] = int(state.attack_state.get("evolve_iter", 0)) + 1

    def _next_proxy_batch(self, device: str) -> torch.Tensor:
        try:
            x_real, _ = next(self.proxy_iter)
        except StopIteration:
            self.proxy_iter = iter(self.proxy_loader)
            x_real, _ = next(self.proxy_iter)
        return x_real.to(device)

    def _pretrain_gan(self, device: str) -> None:
        for _ in range(self.pretrain_steps):
            real_x = self._next_proxy_batch(device)
            z = torch.randn(real_x.size(0), self.noise_dim, device=device)
            fake_x = self.generator(z)

            self.dis_optimizer.zero_grad()
            real_logits = self.discriminator(real_x)
            fake_logits = self.discriminator(fake_x.detach())
            real_labels = torch.ones_like(real_logits)
            fake_labels = torch.zeros_like(fake_logits)
            loss_d = F.binary_cross_entropy_with_logits(real_logits, real_labels)
            loss_d += F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
            loss_d.backward()
            self.dis_optimizer.step()

            self.gen_optimizer.zero_grad()
            fake_logits = self.discriminator(fake_x)
            loss_g = F.binary_cross_entropy_with_logits(fake_logits, real_labels)
            loss_g.backward()
            self.gen_optimizer.step()

    def _train_substitute(self, state: BenchmarkState) -> None:
        query_x = state.attack_state["query_data_x"]
        query_y = state.attack_state["query_data_y"]
        if len(query_x) == 0:
            return

        x_all = torch.cat(query_x, dim=0)
        y_all = torch.cat(query_y, dim=0)

        class QueryDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        loader = torch.utils.data.DataLoader(
            QueryDataset(x_all, y_all),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        device = state.metadata.get("device", "cpu")
        if self.substitute is None:
            sub_config = state.metadata.get("substitute_config", {})
            opt_params = sub_config.get("optimizer", {})
            self.substitute = create_substitute(
                arch=sub_config.get("arch", "resnet18"),
                num_classes=self.num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
            ).to(device)
            self.substitute_optimizer = optim.SGD(
                self.substitute.parameters(),
                lr=float(opt_params.get("lr", self.substitute_lr)),
                momentum=float(opt_params.get("momentum", 0.9)),
                weight_decay=float(opt_params.get("weight_decay", 5e-4))
            )

        self.substitute.train()
        output_mode = self.config.get("output_mode", "soft_prob")
        victim_config = state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)
        
        epochs = max(1, int(self.substitute_epochs))
        for _ in range(epochs):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                # Normalize images for substitute
                x_batch = (x_batch - norm_mean) / norm_std
                
                self.substitute_optimizer.zero_grad()
                logits = self.substitute(x_batch)
                if output_mode == "soft_prob":
                    y_batch = y_batch.to(device)
                    loss = F.kl_div(
                        F.log_softmax(logits, dim=1), y_batch, reduction="batchmean"
                    )
                else:
                    y_batch = y_batch.long().to(device)
                    loss = F.cross_entropy(logits, y_batch)
                loss.backward()
                self.substitute_optimizer.step()

        state.attack_state["substitute"] = self.substitute
