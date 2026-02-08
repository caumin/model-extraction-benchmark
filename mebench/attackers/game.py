"""GAME (Generative-Based Adaptive Model Extraction) attack."""

from typing import Dict, Any, List, Tuple, Optional, Callable
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.models.gan import DCGANGenerator, DCGANDiscriminator, ACGANGenerator, ACGANDiscriminator
from mebench.models.substitute_factory import create_substitute
from mebench.data.loaders import create_dataloader
from mebench.utils.dataloader import load_pool_to_memory


class GAME(AttackRunner):
    """GAME with shared generator/discriminator and adaptive losses."""

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        self.batch_size = int(config.get("batch_size", 128))
        self.student_lr = float(config.get("student_lr", 0.1))
        self.generator_lr = float(config.get("generator_lr", 2e-4))
        self.discriminator_lr = float(config.get("discriminator_lr", 2e-4))
        self.noise_dim = int(config.get("noise_dim", 100))
        self.num_classes = int(
            state.metadata.get("num_classes")
            or config.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        self.base_channels = int(config.get("base_channels", 64))
        self.acs_strategy = config.get("acs_strategy", "uncertainty")
        self.acs_probe_size = int(config.get("acs_probe_size", 0))

        self.beta1 = float(config.get("beta1", 1.0)) # l_res
        self.beta2 = float(config.get("beta2", 1.0)) # l_bou
        self.beta3 = float(config.get("beta3", 0.0)) # l_adv
        self.beta4 = float(config.get("beta4", 1.0)) # l_dif

        # TDL: Training Discriminator and Generator with proxy data.
        # Paper implies iterative training. default to 20 epochs/steps.
        self.tdl_steps = int(config.get("tdl_steps", 20))
        self.agu_steps = int(config.get("agu_steps", 1))
        self.gmd_steps = int(config.get("gmd_steps", 1))
        self.use_acgan = bool(config.get("use_acgan", True))

        self.generator: nn.Module | None = None
        self.discriminator: nn.Module | None = None
        self.student: nn.Module | None = None
        self.generator_optimizer: optim.Optimizer | None = None
        self.discriminator_optimizer: optim.Optimizer | None = None
        self.student_optimizer: optim.Optimizer | None = None
        self.proxy_data: torch.Tensor | None = None
        self.tdl_done = False
        self._ctx: Optional[BenchmarkContext] = None
        self._query_fn: Optional[Callable[..., OracleOutput]] = None
        self._pending_query_k: Optional[int] = None

        self._initialize_state(state)

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        self._ctx = ctx
        self._query_fn = ctx.query
        device = self.state.metadata.get("device", "cpu")
        total_budget = self.state.budget_remaining
        pbar = self._create_progress_bar(total_budget, "[GAME] Extracting")
        
        last_eval_queries = 0
        eval_interval = total_budget // 10

        while ctx.budget_remaining > 0:
            step_size = self._default_step_size(ctx)
            x_query, meta = self._select_query_batch(step_size, self.state)
            if x_query.size(0) == 0:
                break
            oracle_output = ctx.query(x_query, meta=meta)
            self._handle_oracle_output(x_query, meta, oracle_output, self.state)
            pbar.update(x_query.size(0))
            
            # Periodic evaluation
            queries_done = total_budget - ctx.budget_remaining
            if queries_done - last_eval_queries >= eval_interval:
                self._evaluate_current_substitute(self.student, device)
                last_eval_queries = queries_done
                
        pbar.close()
        self._ctx = None
        self._query_fn = None

    def _select_query_batch(self, k: int, state: BenchmarkState) -> tuple[torch.Tensor, dict]:
        self._init_models(state)
        device = state.metadata.get("device", "cpu")
        self._pending_query_k = k
        class_probs = self._compute_class_distribution(state, device)
        self._pending_query_k = None
        class_probs = torch.nan_to_num(class_probs, nan=1.0 / self.num_classes)
        class_probs = torch.clamp(class_probs, min=1e-9)
        class_probs = class_probs / class_probs.sum()

        if self._ctx is not None:
            k = min(k, self._ctx.budget_remaining)
            if k <= 0:
                input_shape = state.metadata.get("input_shape", (3, 32, 32))
                return torch.empty((0, *input_shape), device=device), {"acs_probs": class_probs.cpu()}

        z = torch.randn(k, self.noise_dim, device=device)
        y_g = torch.multinomial(class_probs, k, replacement=True)
        with torch.no_grad():
            x = self.generator(z, y_g)

        x_query = x * 0.5 + 0.5

        meta = {
            "generator_step": state.attack_state["step"],
            "synthetic": True,
            "z": z.cpu(),
            "y_g": y_g.cpu(),
            "acs_probs": class_probs.cpu(),
        }
        return x_query, meta

    def _handle_oracle_output(
        self,
        x_query: torch.Tensor,
        meta: dict,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if self.generator is None or self.discriminator is None or self.student is None:
            return

        device = x_query.device
        if oracle_output.kind == "soft_prob":
            victim_probs = oracle_output.y.to(device)
        else:
            victim_probs = F.one_hot(oracle_output.y, num_classes=self.num_classes).float().to(device)

        self._update_victim_stats(state, victim_probs, meta.get("y_g"))

        state.attack_state["last_victim_probs"] = victim_probs.detach().cpu()
        self._agu_phase(x_query, victim_probs, device, meta.get("z"), meta.get("y_g"))
        self._gmd_phase(x_query, victim_probs)

        state.attack_state["step"] += 1
        state.attack_state["substitute"] = self.student

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state["step"] = 0
        state.attack_state["victim_class_avg_prob"] = torch.full(
            (self.num_classes, self.num_classes),
            1.0 / self.num_classes,
        )
        state.attack_state["victim_class_counts"] = torch.zeros(self.num_classes)

    def _init_models(self, state: BenchmarkState) -> None:
        device = state.metadata.get("device", "cpu")
        if self.generator is None:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            
            # [P0 ARCHITECTURE ENFORCEMENT] GAME requires ACGAN with dropout
            if self.use_acgan:
                self.generator = ACGANGenerator(
                    noise_dim=self.noise_dim,
                    output_channels=int(self.config.get("output_channels", input_shape[0])),
                    base_channels=self.base_channels,
                    num_classes=self.num_classes,
                    output_size=int(input_shape[1]),
                    dropout_prob=0.25,  # Paper-mandated dropout
                ).to(device)
            else:
                self.generator = DCGANGenerator(
                    noise_dim=self.noise_dim,
                    output_channels=int(self.config.get("output_channels", input_shape[0])),
                    base_channels=self.base_channels,
                    num_classes=self.num_classes,
                    output_size=int(input_shape[1]),
                ).to(device)
                
            self.generator_optimizer = optim.Adam(
                self.generator.parameters(), lr=self.generator_lr, betas=(0.5, 0.999)
            )

        if self.discriminator is None:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            
            # [P0 ARCHITECTURE ENFORCEMENT] GAME requires ACGAN with dropout
            if self.use_acgan:
                self.discriminator = ACGANDiscriminator(
                    input_channels=int(self.config.get("output_channels", input_shape[0])),
                    base_channels=self.base_channels,
                    num_classes=self.num_classes,
                    input_size=int(input_shape[1]),
                    dropout_prob=0.25,  # Paper-mandated dropout
                ).to(device)
            else:
                self.discriminator = DCGANDiscriminator(
                    input_channels=int(self.config.get("output_channels", input_shape[0])),
                    base_channels=self.base_channels,
                    num_classes=self.num_classes if self.use_acgan else None,
                    input_size=int(input_shape[1]),
                ).to(device)
                
            self.discriminator_optimizer = optim.Adam(
                self.discriminator.parameters(), lr=self.discriminator_lr, betas=(0.5, 0.999)
            )

        if self.student is None:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            sub_config = state.metadata.get("substitute_config", {})
            opt_params = sub_config.get("optimizer", {})
            
            arch = sub_config.get("arch") or self.config.get("student_arch", "resnet18-8x")
            width_mult = int(sub_config.get("width_mult", 1))
            dropout_prob = float(sub_config.get("dropout_prob", 0.0))
            self.student = create_substitute(
                arch=arch,
                num_classes=self.num_classes,
                input_channels=int(input_shape[0]),
                width_mult=width_mult,
                dropout_prob=dropout_prob,
            ).to(device)
            self.student_optimizer = optim.SGD(
                self.student.parameters(),
                lr=float(opt_params.get("lr", self.student_lr)),
                momentum=float(opt_params.get("momentum", 0.9)),
                weight_decay=float(opt_params.get("weight_decay", 5e-4))
            )

        if self.proxy_data is None:
            proxy_config = self.config.get("attack", {}).get("proxy_dataset")
            if proxy_config is None:
                proxy_config = self.config.get("proxy_dataset")
            if proxy_config is None:
                # Fallback to surrogate if not specified (Track B)
                proxy_config = state.metadata.get("dataset_config", {}).copy()
                proxy_config["data_mode"] = "surrogate"
                # Ensure we have a valid surrogate config
                if "surrogate_name" not in proxy_config:
                    # Try to infer or fail
                    pass
            
            if proxy_config:
                self.proxy_data = load_pool_to_memory(
                    proxy_config,
                    device=device,
                    desc="[GAME] Caching proxy data",
                    max_samples=100_000,
                )

    def _next_proxy_batch(self, device: str) -> torch.Tensor:
        if self.proxy_data is None or self.proxy_data.size(0) == 0:
            # Fallback if no proxy data (should not happen if config correct)
            return torch.randn(self.batch_size, 3, 32, 32, device=device)
            
        indices = torch.randint(0, self.proxy_data.size(0), (self.batch_size,), device=self.proxy_data.device)
        return self.proxy_data[indices].to(device)

    def _compute_class_distribution(self, state: BenchmarkState, device: str) -> torch.Tensor:
        """Compute class distribution from FRESH victim queries for ACS deviation.
        
        [P0 FIX] Paper requires fresh victim queries for ACS deviation, not cached stats.
        """
        if self.student is None or self.generator is None:
            return torch.full((self.num_classes,), 1.0 / self.num_classes, device=device)
        victim_config = self.state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)
        def _norm(img):
            return (img * 0.5 + 0.5 - norm_mean) / norm_std

        if self.acs_strategy != "deviation":
            # Uncertainty-based selection (default): Prefer classes where student is uncertain
            z = torch.randn(self.num_classes, self.noise_dim, device=device)
            class_ids = torch.arange(self.num_classes, device=device)
            with torch.no_grad():
                # We need ACGAN to control class generation. If DCGAN, class_ids ignored.
                # GAME assumes ACGAN for class-conditional generation.
                x_gen_raw = self.generator(z, class_ids)
                x_gen = x_gen_raw * 0.5 + 0.5
                student_logits = self.student(_norm(x_gen))
                student_probs = F.softmax(student_logits, dim=1)
            
            # Uncertainty: 1 - max_prob (or entropy)
            # Higher uncertainty -> Higher selection probability
            score = 1.0 - student_probs.max(dim=1).values
        else:
            # Deviation-based selection: Prefer classes where Student differs from Victim
            if self._query_fn is None:
                self.logger.warning("GAME ACS deviation requires oracle query; returning uniform distribution.")
                return torch.full((self.num_classes,), 1.0 / self.num_classes, device=device)
            
            budget_remaining = None
            if self._ctx is not None:
                budget_remaining = int(self._ctx.budget_remaining)

            probe_budget = self.num_classes
            if self.acs_probe_size > 0:
                probe_budget = min(self.acs_probe_size, self.num_classes)
            if budget_remaining is not None and self._pending_query_k is not None:
                probe_budget = min(probe_budget, max(0, budget_remaining - int(self._pending_query_k)))

            if probe_budget <= 0:
                self.logger.warning("GAME ACS deviation probe skipped due to budget constraints.")
                return torch.full((self.num_classes,), 1.0 / self.num_classes, device=device)

            if probe_budget < self.num_classes:
                class_ids = torch.randperm(self.num_classes, device=device)[:probe_budget]
            else:
                class_ids = torch.arange(self.num_classes, device=device)

            z = torch.randn(class_ids.size(0), self.noise_dim, device=device)
            with torch.no_grad():
                x_gen_raw = self.generator(z, class_ids)
                x_gen = x_gen_raw * 0.5 + 0.5
                student_logits = self.student(_norm(x_gen))
                student_probs = F.softmax(student_logits, dim=1)

            x_query = x_gen # already [0,1]
            oracle_output = self._query_fn(x_query, meta={"acs_probe": True, "y_g": class_ids.detach().cpu()})
            if oracle_output.kind == "soft_prob":
                victim_probs = oracle_output.y.to(device)
            else:
                victim_probs = F.one_hot(oracle_output.y, num_classes=self.num_classes).float().to(device)

            self._update_victim_stats(state, victim_probs, class_ids)
            
            # KL(S || V) approximation
            s_log = torch.log(student_probs + 1e-10)
            v_log = torch.log(victim_probs + 1e-10)
            # We want to select classes with HIGH disagreement
            kl_div = (student_probs * (s_log - v_log)).sum(dim=1)
            
            # Map probe scores back to full class vector
            score = torch.full((self.num_classes,), 1e-6, device=device)
            score[class_ids] = kl_div

        score = score - score.min()
        score = score + 1e-6
        return score / score.sum()

    def _update_victim_stats(self, state: BenchmarkState, probs: torch.Tensor, y_g: Optional[torch.Tensor]) -> None:
        """Update running statistics of victim class probabilities per generated class."""
        if y_g is None:
            return
            
        # Update counts
        # This is a simplified online update for mean vectors per class
        for i, c in enumerate(y_g):
            c_idx = int(c.item())
            if c_idx >= self.num_classes: continue
            
            current_avg = state.attack_state["victim_class_avg_prob"][c_idx].to(probs.device)
            current_n = state.attack_state["victim_class_counts"][c_idx].item()
            
            new_n = current_n + 1
            new_avg = (current_avg * current_n + probs[i]) / new_n
            
            state.attack_state["victim_class_avg_prob"][c_idx] = new_avg
            state.attack_state["victim_class_counts"][c_idx] = new_n

    def _agu_phase(
        self, 
        x_query: torch.Tensor, 
        victim_probs: torch.Tensor, 
        device: str, 
        z: Optional[torch.Tensor],
        y_g: Optional[torch.Tensor]
    ) -> None:
        """Adversarial Generator Update (AGU)."""
        if z is None or y_g is None:
            return

        for _ in range(self.agu_steps):
            # 1. Train Discriminator
            real_x = self._next_proxy_batch(device)
            self.discriminator_optimizer.zero_grad()
            
            # Re-generate fake batch to detach from previous graph
            fake_x_raw = self.generator(z.to(device), y_g.to(device))
            fake_x = fake_x_raw * 0.5 + 0.5
            
            real_validity, real_label = self.discriminator(real_x)
            fake_validity, fake_label = self.discriminator(fake_x.detach())
            
            # Loss D: Real/Fake + Aux Label (ACGAN)
            d_loss_real = F.binary_cross_entropy_with_logits(real_validity, torch.ones_like(real_validity))
            d_loss_fake = F.binary_cross_entropy_with_logits(fake_validity, torch.zeros_like(fake_validity))
            d_loss_cls = 0.0
            if self.use_acgan:
                d_loss_cls = F.cross_entropy(fake_label, y_g.to(device))
                
            d_loss = d_loss_real + d_loss_fake + d_loss_cls
            d_loss.backward()
            self.discriminator_optimizer.step()
            
            # 2. Train Generator
            self.generator_optimizer.zero_grad()
            fake_validity, fake_label = self.discriminator(fake_x)
            
            # Loss G: Fool D + Class Control + Victim Agreement (Boundary)
            g_loss_adv = F.binary_cross_entropy_with_logits(fake_validity, torch.ones_like(fake_validity))
            g_loss_cls = 0.0
            if self.use_acgan:
                g_loss_cls = F.cross_entropy(fake_label, y_g.to(device))
            
            # Boundary Loss (Eq 8): Minimize entropy of victim predictions on generated samples
            # "Samples near decision boundary" -> High entropy
            # But usually we want samples where victim is UNCERTAIN (near boundary).
            # So we should MAXIMIZE entropy (minimize negative entropy).
            # Using self.beta2 * (victim_entropy)
            victim_entropy = -(victim_probs * torch.log(victim_probs + 1e-10)).sum(dim=1).mean()
            # If beta2 > 0, we minimize entropy (make victim confident? No, that's evasion).
            # For extraction, we want informative samples (near boundary) -> Maximize Entropy.
            # So loss term should be -Entropy.
            # Let's assume standard Active Learning heuristic: Maximize Entropy.
            g_loss_bou = -victim_entropy
            
            g_loss = self.beta3 * g_loss_adv + g_loss_cls + self.beta2 * g_loss_bou
            g_loss.backward()
            self.generator_optimizer.step()

    def _gmd_phase(self, x_query: torch.Tensor, victim_probs: torch.Tensor) -> None:
        """Gradient Maximization Discrepancy (GMD) / Student Training."""
        device = x_query.device
        
        # Train Student
        self.student.train()
        self.student_optimizer.zero_grad()
        
        victim_config = self.state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)
        
        student_in = (x_query - norm_mean) / norm_std
        student_logits = self.student(student_in)
        student_probs = F.softmax(student_logits, dim=1)
        
        # L_res: Response Loss (Distillation)
        # Minimize KL(S || V)
        loss_res = F.kl_div(torch.log(student_probs + 1e-10), victim_probs, reduction="batchmean")
        
        # L_dif: Diff Loss (make S different from V?)
        # Wait, GMD is usually for Generator.
        # In GAME paper, S is trained to minimize loss_res.
        # G is trained to maximize discrepancy.
        # This function is called _gmd_phase but actually trains STUDENT.
        # So it should minimize match loss.
        
        loss = self.beta1 * loss_res
        loss.backward()
        self.student_optimizer.step()
