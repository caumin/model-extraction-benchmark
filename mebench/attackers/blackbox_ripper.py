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

        # Paper: population 30, elite 10, max 10 iterations per image.
        self.noise_dim = int(config.get("noise_dim", 128))
        self.latent_bound = float(config.get("latent_bound", 3.0))
        self.population_size = int(config.get("population_size", 30))
        self.elite_size = int(config.get("elite_size", 10))
        # mutation_scale removed/fixed to 1.0 later for strict N(0,1).
        # self.mutation_scale = float(config.get("mutation_scale", 1.0)) 
        self.fitness_threshold = float(config.get("fitness_threshold", 0.02))
        self.max_evolve_iters = int(config.get("max_evolve_iters", 10))
        
        # Parallel slots limit for budget fairness
        self.max_slots_per_round = int(config.get("max_slots_per_round", 1))
        
        total_budget = int(state.metadata.get("max_budget", 10000))
        # train_every removed in favor of end-of-budget training.
        # self.train_every = int(config.get("train_every", max(256, total_budget // 10)))
        
        self.pretrain_steps = int(config.get("pretrain_steps", 100))
        self.substitute_lr = float(config.get("substitute_lr", 0.01))
        
        # [STRICT] Paper specifies training for 200 epochs.
        self.substitute_epochs = int(config.get("substitute_epochs", 200))
        
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
        state.attack_state["active_populations"] = {}  # index -> population tensor
        state.attack_state["active_targets"] = {}      # index -> target label
        state.attack_state["evolve_iters"] = {}        # index -> iteration count
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []
        state.attack_state["substitute"] = None
        state.attack_state["total_queries"] = 0
        state.attack_state["substitute_trained"] = False

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
        
        active_populations = state.attack_state["active_populations"]
        active_targets = state.attack_state["active_targets"]
        evolve_iters = state.attack_state["evolve_iters"]
        
        # Calculate how many full populations (slots) we can afford with budget k.
        # k is the query budget for THIS call.
        pop_size = self.population_size
        slots_budget = k // pop_size
        
        if slots_budget == 0:
            # Not enough budget for even one full population evaluation.
            # Return empty to signal atomic constraint. Do NOT return noise.
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            return QueryBatch(
                x=torch.empty((0, *input_shape), device=device),
                meta={"slot_ids": [], "batch_size": 0}
            )
            
        # Cap parallel slots to ensure fair comparison
        slots_to_process = min(slots_budget, self.max_slots_per_round)
        
        # Fill/Refresh active slots up to capacity
        current_active_keys = list(active_populations.keys())
        needed = max(0, slots_to_process - len(current_active_keys))
        
        next_id = max(current_active_keys, default=-1) + 1
        
        for _ in range(needed):
            # Initialize new population for a new slot
            u = float(self.latent_bound)
            pop = (torch.rand(self.population_size, self.noise_dim, device=device) * 2.0 - 1.0) * u
            target = torch.randint(0, self.num_classes, (1,), device=device).item()
            
            active_populations[next_id] = pop
            active_targets[next_id] = target
            evolve_iters[next_id] = 0
            next_id += 1
            
        # Select exactly slots_to_process slots to query
        # Since we just filled 'needed', we have at least 'slots_to_process' slots.
        # We take the first available ones.
        # Note: dict keys are insertion ordered in modern python, but list(...) is safer.
        final_batch_ids = list(active_populations.keys())[:slots_to_process]
        
        batch_pop_list = []
        meta_indices = []
        
        for slot_id in final_batch_ids:
            pop = active_populations[slot_id].to(device)
            batch_pop_list.append(pop)
            meta_indices.extend([slot_id] * self.population_size)
            
        x_raw = self.generator(torch.cat(batch_pop_list, dim=0))
        x = x_raw * 0.5 + 0.5
        
        meta = {
            "slot_ids": meta_indices, # Defines which slot each image belongs to
            "batch_size": len(meta_indices)
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

        slot_ids = query_batch.meta.get("slot_ids", [])
        if not slot_ids:
            return

        device = state.metadata.get("device", "cpu")
        active_populations = state.attack_state["active_populations"]
        active_targets = state.attack_state["active_targets"]
        evolve_iters = state.attack_state["evolve_iters"]
        
        if oracle_output.kind == "soft_prob":
            all_probs = oracle_output.y.to(device)
        else:
            all_probs = F.one_hot(
                oracle_output.y, num_classes=self.num_classes
            ).float().to(device)

        # Group by slot_id
        unique_slots = sorted(list(set(slot_ids)))
        
        # We need to map linear index in batch to slot
        # Robust way: iterate unique slots and use list comprehension masking
        
        start_idx = 0
        for slot_id in unique_slots:
            # Find all indices in the batch belonging to this slot
            # Note: We assume propose() keeps them contiguous, but we should be robust.
            # However, batch is ordered by slot_ids metadata.
            # slot_ids is a list matching x_batch order.
            
            # Efficient masking:
            indices = [i for i, sid in enumerate(slot_ids) if sid == slot_id]
            count = len(indices)
            
            # Strict check: Must process exactly K teacher outputs per slot
            if count != self.population_size:
                raise RuntimeError(f"Expected {self.population_size} outputs for slot {slot_id}, got {count}")
            
            # Extract probs for this slot
            # If contiguous (which it is from propose), slice is faster.
            # If we want to be paranoid-robust:
            # probs = all_probs[indices]
            
            # Let's check contiguous assumption to be safe, but fallback to indexing
            is_contiguous = (indices == list(range(indices[0], indices[0]+count)))
            if is_contiguous:
                probs = all_probs[indices[0] : indices[0]+count]
            else:
                probs = all_probs[indices]
                
            current_pop = active_populations[slot_id].to(device)
            target_cls = active_targets[slot_id]
            
            # Evolution Step
            best_z, best_score, new_pop, best_idx_in_pop = self._evolve_single_slot(
                current_pop, target_cls, probs
            )
            
            # Update state
            active_populations[slot_id] = new_pop.cpu()
            evolve_iters[slot_id] += 1
            
            # Check convergence or max iters
            if evolve_iters[slot_id] >= self.max_evolve_iters or best_score >= -self.fitness_threshold:
                # Add best sample to training data
                with torch.no_grad():
                    best_img = self.generator(best_z.unsqueeze(0)) * 0.5 + 0.5
                    
                state.attack_state["query_data_x"].append(best_img.cpu())
                
                # Use best_idx_in_pop to get exactly the probability vector corresponding to best_z
                best_prob = probs[best_idx_in_pop]
                state.attack_state["query_data_y"].append(best_prob.cpu().unsqueeze(0))
                
                # Remove from active set to make room for new
                del active_populations[slot_id]
                del active_targets[slot_id]
                del evolve_iters[slot_id]
            
            # Note: start_idx logic not needed if using indices map

        # Policy A: Train once at end of budget.
        # Check if we can afford another full population step.
        max_budget = int(state.metadata.get("max_budget", 0))
        remaining = max_budget - int(state.query_count)
        
        # If remaining budget < population_size, we can't do another step.
        # So we trigger final training.
        if (not state.attack_state.get("substitute_trained")) and (remaining < self.population_size):
            self._train_substitute(state)
            state.attack_state["substitute_trained"] = True

    def _evolve_single_slot(
        self,
        population: torch.Tensor,
        target_cls: int,
        victim_probs: torch.Tensor
    ):
        device = population.device
        target_onehot = F.one_hot(torch.tensor(target_cls), num_classes=self.num_classes).float().to(device)
        
        # Fitness = -Objective (Minimize Objective)
        # Eq. (2) Objective: L(x) = || F(x) - t ||^2  (Sum of squared differences)
        # Paper usually implies L2 norm squared.
        diff = victim_probs - target_onehot
        mse_sum = (diff * diff).sum(dim=1)
        fitness = -mse_sum
        
        # Select Elites
        elite_count = min(self.elite_size, population.size(0))
        topk = torch.topk(fitness, elite_count)
        elite_indices = topk.indices
        best_score = topk.values[0].item() # This is -mse_sum
        best_idx_in_pop = elite_indices[0].item()
        
        elites = population[elite_indices]
        best_z = elites[0]
        
        # Mutation (Algorithm 1 Step 6)
        # "Generate offspring by adding random noise to the elite samples."
        # Implementation: Sample (K-k) parents from elites uniformly with replacement.
        needed = self.population_size - elite_count
        
        offspring = []
        if needed > 0:
            parent_indices = torch.randint(0, elite_count, (needed,), device=device)
            parents = elites[parent_indices]
            
            # Mutate all parents: child = parent + noise
            # Note: Paper says "adding random noise". 
            # Freeze mutation to strictly N(0,1) for reproducibility/fairness.
            # mutation_scale controls the magnitude (sigma).
            # noise = torch.randn_like(parents) * self.mutation_scale
            noise = torch.randn_like(parents)
            offspring_tensor = parents + noise
            
            new_population = torch.cat([elites, offspring_tensor], dim=0)
        else:
            new_population = elites
            
        # Clip is removed for strict fidelity to paper text unless code proves otherwise.
        # Paper doesn't explicitly mention clamping latent vector after mutation.
        # However, if latent_bound is used for initialization, it might be implied?
        # "Zero Tolerance": If not in text/algorithm, don't add it.
        # if self.latent_bound > 0:
        #     new_population = new_population.clamp(-self.latent_bound, self.latent_bound)
            
        return best_z, best_score, new_population, best_idx_in_pop

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
            fake_x = self.generator(z) * 0.5 + 0.5

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
        # Strict fidelity: Train from scratch for 200 epochs on current data.
        # This aligns with "Train the substitute for 200 epochs" on the dataset.
        # Repeated training on checkpoints ensures Track B is valid.
        
        # Reset model to scratch?
        # If we continue training, it's > 200 epochs effectively.
        # Paper implies "Offline" training.
        # So we should reset.
        
        device = state.metadata.get("device", "cpu")
        sub_config = state.metadata.get("substitute_config", {})
        opt_params = sub_config.get("optimizer", {})
        
        self.substitute = create_substitute(
            arch=sub_config.get("arch", "resnet18"),
            num_classes=self.num_classes,
            input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
        ).to(device)
        
        self.substitute_optimizer = optim.Adam(
            self.substitute.parameters(),
            lr=float(opt_params.get("lr", self.substitute_lr)),
        )

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

        self.substitute.train()
        output_mode = self.config.get("output_mode", "soft_prob")
        victim_config = state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)
        
        # Paper: "We train the substitute model for 200 epochs using the Adam optimizer."
        # Setup implies batch size 64.
        
        train_loader = torch.utils.data.DataLoader(
            QueryDataset(x_all, y_all),
            batch_size=64, # Explicitly 64 as per paper setup
            shuffle=True,
            num_workers=0,
        )
        
        epochs = max(1, int(self.substitute_epochs))
        for _ in range(epochs):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                x_batch = (x_batch - norm_mean) / norm_std
                
                self.substitute_optimizer.zero_grad()
                logits = self.substitute(x_batch)
                if output_mode == "soft_prob":
                    y_batch = y_batch.to(device)
                    # Use exact Cross-Entropy form for soft labels as per paper logic
                    # H(p, q) = -sum p(x) log q(x)
                    log_probs = F.log_softmax(logits, dim=1)
                    # Note: y_batch are the target probabilities (teacher)
                    loss = -(y_batch * log_probs).sum(dim=1).mean()
                else:
                    y_batch = y_batch.long().to(device)
                    loss = F.cross_entropy(logits, y_batch)
                loss.backward()
                self.substitute_optimizer.step()

        state.attack_state["substitute"] = self.substitute
