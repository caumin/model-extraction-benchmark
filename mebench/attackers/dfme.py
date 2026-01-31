"""DFME (Data-Free Model Extraction) attack implementation."""

from typing import Dict, Any, Tuple, List, Optional
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.models.gan import DCGANGenerator, DFMEGenerator
from mebench.models.substitute_factory import create_substitute


class DFME(AttackRunner):
    """Data-Free Model Extraction via generative adversarial training.

    Algorithm loop (min-max game):
    1. G generates queries: x = G(z) where z ~ N(0, 1)
    2. S learns to mimic V: minimize disagreement (L1 loss)
    3. G learns to maximize disagreement between S and V
    4. Use zeroth-order gradient estimation for G (V is black-box)
    5. Repeat until query budget exhausted

    Hyperparameters (from AGENTS.md):
    - Batch size: 256
    - Student optimizer: SGD, lr=0.1, weight_decay=5e-4
    - Generator optimizer: Adam, lr=5e-4
    - Steps: n_G=1, n_S=5
    - Gradient approx: m=1, epsilon=1e-3
    """

    def __init__(self, config: dict, state: BenchmarkState):
        """Initialize DFME attack.

        Args:
            config: Attack configuration
            state: Global benchmark state
        """
        super().__init__(config, state)

        # Hyperparameters (from AGENTS.md)
        self.batch_size = int(config.get("batch_size", 256))
        self.student_lr = float(config.get("student_lr", 0.01))
        self.student_weight_decay = float(config.get("student_weight_decay", 5e-4))
        # Official DFME repo commonly uses lr_G=1e-4, but paper specifies 5e-4.
        self.generator_lr = float(config.get("generator_lr", 5e-4))
        self.lr_steps = config.get("lr_steps", [0.1, 0.3, 0.5])
        self.lr_scale = float(config.get("lr_scale", 0.3))
        self.n_g_steps = int(config.get("n_g_steps", 1))  # Generator steps
        self.n_s_steps = int(config.get("n_s_steps", 5))  # Student steps
        self.grad_approx_m = int(config.get("grad_approx_m", 1))  # Random directions
        self.grad_approx_epsilon = float(config.get("grad_approx_epsilon", 1e-3))

        # Noise distribution
        self.noise_dim = int(config.get("noise_dim", 100))  # z ~ N(0, 1)

        # Initialize models
        self.student = None  # Will be initialized on selection
        self.generator = None  # Will be initialized on selection
        self.student_optimizer = None
        self.generator_optimizer = None
        self._x_shape = None

        # Initialize attack state
        self._initialize_state(state)

    def _initialize_state(self, state: BenchmarkState) -> None:
        """Initialize attack-specific state.

        Args:
            state: Global benchmark state to update
        """
        # Training progress
        state.attack_state["step"] = 0
        state.attack_state["lr_step_index"] = 0

    def _create_student(self, input_shape: tuple) -> nn.Module:
        """Create student model.

        Args:
            input_shape: Input shape (C, H, W)

        Returns:
            Student model
        """
        sub_config = self.state.metadata.get("substitute_config", {})
        arch = sub_config.get("arch", "resnet18-8x")
        num_classes = int(
            self.state.metadata.get("num_classes")
            or self.config.get("num_classes")
            or self.state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        input_channels = int(input_shape[0])
        return create_substitute(
            arch=arch,
            num_classes=num_classes,
            input_channels=input_channels,
        )

    def _create_generator(self) -> nn.Module:
        """Create generator model."""
        gen_type = str(self.config.get("generator_type", "dfme")).lower()
        input_shape = self.state.metadata.get("input_shape", (3, 32, 32))
        output_size = int(input_shape[1])
        output_channels = int(self.config.get("output_channels", input_shape[0]))
        base_channels = int(self.config.get("base_channels", 128))
        if gen_type in {"dfme", "upsample_conv"}:
            return DFMEGenerator(
                noise_dim=self.noise_dim,
                output_channels=output_channels,
                base_channels=base_channels,
                output_size=output_size,
            )
        return DCGANGenerator(
            noise_dim=self.noise_dim,
            output_channels=output_channels,
            base_channels=int(self.config.get("base_channels", 64)),
            num_classes=None,
            output_size=output_size,
        )

    def _ensure_models(self, device: str) -> None:
        if self.generator is None:
            self.generator = self._create_generator().to(device)
            self.generator_optimizer = optim.Adam(
                self.generator.parameters(), lr=self.generator_lr
            )
        if self.student is None:
            self.student = self._create_student(self.state.metadata.get("input_shape", (3, 32, 32))).to(device)
            sub_config = self.state.metadata.get("substitute_config", {})
            opt_params = sub_config.get("optimizer", {})
            self.student_optimizer = optim.SGD(
                self.student.parameters(),
                lr=float(opt_params.get("lr", self.student_lr)),
                momentum=float(opt_params.get("momentum", 0.9)),
                weight_decay=float(opt_params.get("weight_decay", self.student_weight_decay)),
            )

    def _get_x_shape(self, device: str) -> torch.Size:
        if self._x_shape is None:
            with torch.no_grad():
                pre_tanh_probe, _ = self.generator(
                    torch.randn(1, self.noise_dim, device=device), return_pre_tanh=True
                )
            self._x_shape = pre_tanh_probe.shape[1:]
        return self._x_shape

    def _to_victim_logits(self, oracle_output: OracleOutput, device: str) -> torch.Tensor:
        if oracle_output.kind == "soft_prob":
            log_probs = torch.log(oracle_output.y.to(device) + 1e-6)
            return log_probs - log_probs.mean(dim=1, keepdim=True)

        num_classes = int(
            self.state.metadata.get("num_classes")
            or self.config.get("num_classes")
            or self.state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        victim_logits = torch.zeros(oracle_output.y.shape[0], num_classes, device=device)
        victim_logits.scatter_(1, oracle_output.y.to(device).unsqueeze(1), 1.0)
        return victim_logits

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        device = self.state.metadata.get("device", "cpu")
        self._ensure_models(device)
        x_shape = self._get_x_shape(device)

        total_budget = ctx.budget_remaining
        pbar = tqdm(total=total_budget, desc="[DFME] Extracting")
        last_eval_queries = 0
        eval_interval = total_budget // 10 # Evaluate 10 times during extraction

        while ctx.budget_remaining > 0:
            # Generator steps
            for _ in range(self.n_g_steps):
                if ctx.budget_remaining <= 0:
                    break

                self.generator.train()
                self.student.eval()
                self.generator_optimizer.zero_grad(set_to_none=True)
                g_updates = 0

                for _ in range(self.batch_size):
                    if ctx.budget_remaining <= 0:
                        break

                    remaining = ctx.budget_remaining
                    if remaining <= 0:
                        break

                    m_eff = 0
                    if remaining > 1:
                        m_eff = min(int(self.grad_approx_m), remaining - 1)

                    z = torch.randn(1, self.noise_dim, device=device)
                    pre_tanh, x_base = self.generator(z, return_pre_tanh=True)

                    dirs = torch.zeros(1, 0, *x_shape, device=device)
                    x_query = x_base
                    if m_eff > 0:
                        dirs = torch.randn(1, m_eff, *x_shape, device=device)
                        norm = dirs.view(1, m_eff, -1).norm(dim=2, keepdim=True) + 1e-12
                        dirs = dirs / norm.view(1, m_eff, 1, 1, 1)
                        pre_tanh_pert = pre_tanh.unsqueeze(1) + self.grad_approx_epsilon * dirs
                        pre_tanh_pert = pre_tanh_pert.view(m_eff, *x_shape)
                        x_pert = torch.tanh(pre_tanh_pert)
                        x_query = torch.cat([x_base, x_pert], dim=0)

                    oracle_output = ctx.query(x_query)
                    pbar.update(x_query.size(0))
                    victim_logits = self._to_victim_logits(oracle_output, device)

                    v_base = victim_logits[:1]
                    v_pert = victim_logits[1:]

                    s_base = self.student(self._normalize_dfme(x_base))
                    loss_base = torch.sum(torch.abs(s_base - v_base), dim=1)

                    grad_est = torch.zeros_like(pre_tanh)
                    if m_eff > 0:
                        with torch.no_grad():
                            s_pert = self.student(self._normalize_dfme(x_query[1:]))
                            loss_pert = torch.sum(torch.abs(s_pert - v_pert), dim=1)

                        loss_diff = (loss_pert - loss_base) / self.grad_approx_epsilon
                        d = pre_tanh.view(pre_tanh.size(0), -1).shape[1]
                        for di in range(m_eff):
                            grad_est += loss_diff[di] * dirs[0, di].unsqueeze(0)
                        grad_est = grad_est / m_eff  # [P0 FIX] Removed dimension scaling as per paper

                    pre_tanh.backward(-grad_est)
                    g_updates += 1

                if g_updates > 0:
                    self.generator_optimizer.step()
                    self._maybe_step_lr(self.state)

# Student steps
            for _ in range(self.n_s_steps):
                if ctx.budget_remaining <= 0:
                    break
                
                n_samples = min(self.batch_size, ctx.budget_remaining)
                if n_samples <= 0:
                    break
                
                # [P0 FIX] Validate budget allocation ratio compliance
                # Paper specifies n_G:n_S = 1:5, validate we maintain this ratio
                total_g_queries = self.n_g_steps * n_samples
                expected_s_steps = max(1, total_g_queries // 5)
                
                if self.n_s_steps != expected_s_steps:
                    self.logger.warning(
                        f"Budget allocation deviation: n_G={self.n_g_steps}, "
                        f"expected n_S={expected_s_steps} for ratio 1:5"
                    )

                z = torch.randn(n_samples, self.noise_dim, device=device)
                _, x_base = self.generator(z, return_pre_tanh=True)

                oracle_output = ctx.query(x_base)
                pbar.update(x_base.size(0))
                victim_logits = self._to_victim_logits(oracle_output, device)

                self.student.train()
                self.student_optimizer.zero_grad(set_to_none=True)
                s_out = self.student(self._normalize_dfme(x_base.detach()))
                loss = torch.mean(torch.sum(torch.abs(s_out - victim_logits), dim=1))
                loss.backward()
                self.student_optimizer.step()

                self._maybe_step_lr(self.state)

            self.state.attack_state["step"] += 1
            self.state.attack_state["substitute"] = self.student
            
            # Periodic evaluation
            queries_done = total_budget - ctx.budget_remaining
            if queries_done - last_eval_queries >= eval_interval:
                # self._evaluate_current_substitute(self.student, device)
                last_eval_queries = queries_done
        
        pbar.close()

    def _select_query_batch(self, k: int, state: BenchmarkState) -> QueryBatch:
        """Select k synthetic queries generated by G.

        Args:
            k: Number of queries to select
            state: Current benchmark state

        Returns:
            QueryBatch with k synthetic queries
        """
        # Initialize models if first time
        device = state.metadata.get("device", "cpu")
        self._ensure_models(device)
        if state.budget_remaining is not None and state.budget_remaining > 0:
            k = min(int(k), int(state.budget_remaining))
        k = int(k)
        if k <= 0:
            raise ValueError("DFME select(k) requires k>0")

        # Contract: selection must return exactly k images.
        remaining = k
        z_list = []
        m_list = []
        x_parts = []
        directions_list = []

        # Probe output shape once.
        with torch.no_grad():
            pre_tanh_probe, _ = self.generator(torch.randn(1, self.noise_dim, device=device), return_pre_tanh=True)
            x_shape = pre_tanh_probe.shape[1:]

        # Total queries needed for one full iteration (n_G + n_S steps)
        # Note: G-step consumes queries for gradient approximation (1 + m) per sample.
        # But here 'k' is the query budget given by the engine.
        # If we want to support true multi-step updates, 'k' should be large enough.
        #
        # Current logic:
        # We pack multiple "steps" into one query batch.
        # Total needed = batch_size * (n_G + n_S)
        # But 'k' is given. So we must adapt.
        #
        # If k >= batch_size * (n_G + n_S):
        #   We can form n_G batches for G-step and n_S batches for S-step.
        # Else:
        #   We just fill 'k' and iterate as much as possible.
        
        remaining = k
        z_list = []
        m_list = []
        x_parts = []
        directions_list = []
        step_types = [] # 'G' or 'S'

        # Probe output shape once.
        with torch.no_grad():
            pre_tanh_probe, _ = self.generator(torch.randn(1, self.noise_dim, device=device), return_pre_tanh=True)
            x_shape = pre_tanh_probe.shape[1:]

        # Planning phase: Determine how many G-batches and S-batches we can form
        # We prioritize G-step batches first, then S-step batches.
        # Standard loop: 1 G-step, then 5 S-steps.
        
        # Calculate cost per batch type
        cost_per_g_sample = 1 + int(self.grad_approx_m)
        cost_per_s_sample = 1
        
        # We want to maintain ratio n_G : n_S batches
        # But we are constrained by 'k' queries.
        
        # Simple strategy: Just generate samples. The handler will split them.
        # To support "fresh data per S-step", we need (n_S * batch_size) samples for S-steps.
        # And (n_G * batch_size * cost_per_g) samples for G-steps.
        
        # Since we just receive 'k' (budget), we produce 'k' queries.
        # The handler will have to deal with whatever amount it gets.
        # BUT, to help the handler, we tag samples as intended for 'G' or 'S'.

        # We assume k is a multiple of batch_size for standard operation,
        # but if k is small (end of budget), we just produce what we can.
        
        # We iterate: 1 G-batch, then n_S S-batches.
        # current_k = 0
        # while current_k < k:
        #      # Can we fit a G-batch?
        #      # Approximate check. We fill sample by sample.
        #      pass

        # while remaining > 0:
            # Decide type: G or S?
            # We follow the cycle: n_G batches for G, then n_S batches for S.
            # But we are generating sample-by-sample (or mini-batch by mini-batch).
            # Let's simplify:
            # We generate heterogeneous batch.
            # We tag each sample with metadata 'purpose': 'G' or 'S'
            
            # To correctly interleave, we need to know where we are in the "virtual" loop.
        # Let's rely on the handler to slice.
            # Here we just produce samples.
            # Wait, for G-step we need perturbation directions. For S-step we don't.
            # So we MUST know the purpose at generation time.
            
            # Heuristic:
            # cycle_cost = (n_G * batch_size * (1+m)) + (n_S * batch_size * 1)
            # We try to complete integer number of cycles.
            
            # Actually, simpler:
            # We just produce G-samples (with perturbations) and S-samples (clean) 
            # in the correct ratio until budget 'k' runs out.
            
            # We need persistent state for this cycle if k is small?
        # Selection is usually large (e.g. 20k queries).
        # The runner calls selection with k.
            
            # Let's execute the schedule loop here
            # For t = 1 to ...
            #   For i = 1 to n_G: generate G-batch
            #   For j = 1 to n_S: generate S-batch
            
            # We break this structured loop into a flat stream of queries.
            
            # break # Replaced by logic below

        # Structured generation loop
        queries_generated = 0
        cycle_idx = 0
        
        while queries_generated < k:
            # 1. Generator Step (n_G batches)
            for _ in range(self.n_g_steps):
                if queries_generated >= k: break
                
                # We need to fill one batch (or as much as possible)
                samples_needed = self.batch_size
                
                for _ in range(samples_needed):
                    if queries_generated >= k: break
                    
                    # One G-sample needs (1 + m) queries
                    m_eff = 0
                    if k - queries_generated > 1:
                         m_eff = min(int(self.grad_approx_m), k - queries_generated - 1)
                    
                    z = torch.randn(1, self.noise_dim, device=device)
                    with torch.no_grad():
                        pre_tanh, x_base = self.generator(z, return_pre_tanh=True)
                    
                    x_parts.append(x_base)
                    z_list.append(z)
                    m_list.append(m_eff)
                    step_types.append(1) # 1 for G-step base
                    
                    queries_generated += 1
                    
                    if m_eff > 0:
                        dirs = torch.randn(1, m_eff, *x_shape, device=device)
                        norm = dirs.view(1, m_eff, -1).norm(dim=2, keepdim=True) + 1e-12
                        dirs = dirs / norm.view(1, m_eff, 1, 1, 1)
                        pre_tanh_pert = pre_tanh.unsqueeze(1) + self.grad_approx_epsilon * dirs
                        pre_tanh_pert = pre_tanh_pert.view(m_eff, *x_shape)
                        x_pert = torch.tanh(pre_tanh_pert)
                        
                        x_parts.append(x_pert)
                        directions_list.append(dirs)
                        queries_generated += m_eff
                        
                        # Add metadata placeholders for perturbed queries (they don't need separate z/m entries in our packing scheme)
        # We only track base z/m. The handler unpacks perturbed samples.
                        # Wait, x_parts grows by (1+m).
                        # z_list grows by 1.
                        # m_list grows by 1.
        # This aligns with handler unpacking logic.
                    else:
                        directions_list.append(torch.zeros(1, 0, *x_shape, device=device))

            # 2. Student Step (n_S batches)
            for _ in range(self.n_s_steps):
                if queries_generated >= k: break
                
                # We need to fill one batch (or as much as possible)
                # S-samples are just base samples, no perturbations needed (m=0)
                samples_needed = self.batch_size
                
                for _ in range(samples_needed):
                    if queries_generated >= k: break
                    
                    z = torch.randn(1, self.noise_dim, device=device)
                    with torch.no_grad():
                        _, x_base = self.generator(z, return_pre_tanh=True)
                        
                    x_parts.append(x_base)
                    z_list.append(z)
                    m_list.append(0) # S-step needs no perturbations
                    step_types.append(2) # 2 for S-step
                    
                    # Placeholder direction
                    directions_list.append(torch.zeros(1, 0, *x_shape, device=device))
                    
                    queries_generated += 1
            
            cycle_idx += 1

        x_all = torch.cat(x_parts, dim=0)
        # Verify packing
        if x_all.size(0) != k:
             # Should not happen with above logic, but safety check
            raise RuntimeError(f"DFME select(k) packing bug: expected {k}, got {x_all.size(0)}")

        max_m = int(self.grad_approx_m)
        n_bases = len(z_list)
        directions = torch.zeros(n_bases, max_m, *x_shape, device=device)
        m_per_base = torch.zeros(n_bases, dtype=torch.long, device=device)
        step_types_tensor = torch.tensor(step_types, dtype=torch.long, device=device)

        for i, m_eff in enumerate(m_list):
            m_per_base[i] = int(m_eff)
            if m_eff > 0:
                directions[i, :m_eff] = directions_list[i][0]

        meta = {
            "generator_step": state.attack_state["step"],
            "synthetic": True,
            "z": torch.cat(z_list, dim=0).detach().cpu(),
            "directions": directions.detach().cpu(),
            "m_per_base": m_per_base.detach().cpu(),
            "step_types": step_types_tensor.detach().cpu(), # 1=G, 2=S
        }

        return QueryBatch(x=x_all, meta=meta)

    def _handle_oracle_output(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        """Update G and S based on oracle response.

        Performs training iterations (n_G and n_S steps) as interleaved in the query batch.

        Args:
            query_batch: The query batch that was sent
            oracle_output: Oracle response
            state: Current benchmark state
        """
        x_all = query_batch.x
        device = x_all.device
        z_cpu = query_batch.meta.get("z")
        directions_cpu = query_batch.meta.get("directions")
        m_per_base_cpu = query_batch.meta.get("m_per_base")
        step_types_cpu = query_batch.meta.get("step_types") # 1=G, 2=S

        if z_cpu is None or directions_cpu is None or m_per_base_cpu is None or step_types_cpu is None:
            return

        z = z_cpu.to(device)
        directions = directions_cpu.to(device)
        m_per_base = m_per_base_cpu.to(device)
        step_types = step_types_cpu.to(device)

        # Convert oracle output to logits if needed
        if oracle_output.kind == "soft_prob":
            log_probs = torch.log(oracle_output.y.to(device) + 1e-6)
            victim_logits = log_probs - log_probs.mean(dim=1, keepdim=True)
        else:
            num_classes = int(
                self.state.metadata.get("num_classes")
                or self.config.get("num_classes")
                or self.state.metadata.get("dataset_config", {}).get("num_classes", 10)
            )
            victim_logits = torch.zeros(oracle_output.y.shape[0], num_classes).to(device)
            victim_logits.scatter_(1, oracle_output.y.to(device).unsqueeze(1), 1.0)

        num_bases = len(z)
        cursor = 0
        base_idx = 0

        while base_idx < num_bases:
            # Determine current step type
            current_type = int(step_types[base_idx].item())
            
            # Collect a batch of bases for this step
            batch_indices = []
            batch_m = []
            batch_z = []
            
            while len(batch_indices) < self.batch_size and base_idx < num_bases:
                if int(step_types[base_idx].item()) != current_type:
                    break
                
                batch_indices.append(base_idx)
                batch_m.append(int(m_per_base[base_idx].item()))
                batch_z.append(z[base_idx])
                
                base_idx += 1
            
            if not batch_indices: break

            # Processing G-step (type 1)
            if current_type == 1:
                self.generator_optimizer.zero_grad()
                self.student.eval() # Student is frozen during G-step
                
                for k_idx, local_m in enumerate(batch_m):
                    local_base_idx = batch_indices[k_idx]
                    local_z = batch_z[k_idx].unsqueeze(0)
                    local_dirs = directions[local_base_idx] # [max_m, ...]
                    
                    # Extract victim outputs
                    v_base = victim_logits[cursor : cursor+1]
                    cursor += 1
                    
                    v_pert = None
                    if local_m > 0:
                        v_pert = victim_logits[cursor : cursor+local_m]
                        cursor += local_m
                    
                    # Compute Gradient Estimation for this single sample
                    # 1. Forward G with z (re-compute graph)
                    pre_tanh, x_base_recon = self.generator(local_z, return_pre_tanh=True)
                    
                    # 2. Student output on base
                    # Paper Eq. 5 disagreement: sum of absolute differences in logits
                    s_base = self.student(self._normalize_dfme(x_base_recon))
                    loss_base = torch.sum(torch.abs(s_base - v_base), dim=1)
                    
                    grad_est = torch.zeros_like(pre_tanh)
                    
                    if local_m > 0:
                        # Corresponds to perturbed samples in flat x_all
                        x_pert_tensor = x_all[cursor-local_m : cursor]
                        
                        with torch.no_grad():
                             # Consistent preprocessing for student
                             s_pert = self.student(self._normalize_dfme(x_pert_tensor))
                             loss_pert = torch.sum(torch.abs(s_pert - v_pert), dim=1)
                        
                        # Calculate diff
                        loss_diff = (loss_pert - loss_base) / self.grad_approx_epsilon
                        
                        # Calculate input dimension d (pre-tanh space)
                        d = pre_tanh.view(pre_tanh.size(0), -1).shape[1]

# Eq. 6: g = (d/m) * sum( loss_diff * dir )
                        for di in range(local_m):
                            grad_est += loss_diff[di] * local_dirs[di].unsqueeze(0)
                        
                        # [P0 FIX] Remove dimension scaling to match run() method
                        # grad_est is raw gradient vector (dimension included), no division by local_m
                        # This matches paper: "no dimension scaling" rule
                    
                    # Gradient Ascent: Maximize disagreement (minimize negative disagreement)
                    pre_tanh.backward(-grad_est)
                
                self.generator_optimizer.step()
            
            # Processing S-step (type 2)
            elif current_type == 2:
                x_batch_list = []
                v_batch_list = []
                
                for k_idx, local_m in enumerate(batch_m):
                    x_batch_list.append(x_all[cursor : cursor+1])
                    v_batch_list.append(victim_logits[cursor : cursor+1])
                    cursor += 1
                
                x_batch = torch.cat(x_batch_list, dim=0)
                v_batch = torch.cat(v_batch_list, dim=0)
                
                self.student.train()
                # Update S
                self.student_optimizer.zero_grad()
                s_out = self.student(self._normalize_dfme(x_batch.detach()))
                # S-step also uses sum of abs differences per Eq. 5
                loss = torch.mean(torch.sum(torch.abs(s_out - v_batch), dim=1))
                loss.backward()
                self.student_optimizer.step()

        # Update step counter
        state.attack_state["step"] += num_bases // self.batch_size
        
        self._maybe_step_lr(state)
        state.attack_state["substitute"] = self.student

        # Update step counter (approximate, since we did multiple steps)
        # We count 1 step per full cycle? Or just increment.
        state.attack_state["step"] += num_bases // self.batch_size # Approximate
        
        self._maybe_step_lr(state)
        state.attack_state["substitute"] = self.student


    def _normalize_dfme(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize images for student model.
        
        Note: Generator outputs [-1, 1], but standard normalization expects [0, 1].
        We shift before normalizing.
        """
        victim_config = self.state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        
        device = x.device
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)
        
        # Shift [-1, 1] -> [0, 1]
        x_01 = (x + 1) / 2
        
        return (x_01 - norm_mean) / norm_std

    def _maybe_step_lr(self, state: BenchmarkState) -> None:
        if self.student_optimizer is None or self.generator_optimizer is None:
            return

        max_budget = state.metadata.get("max_budget")
        if max_budget is None:
            return

        step_index = int(state.attack_state.get("lr_step_index", 0))
        if step_index >= len(self.lr_steps):
            return

        thresholds = [int(step * max_budget) for step in self.lr_steps]
        if state.query_count < thresholds[step_index]:
            return

        for optimizer in [self.student_optimizer, self.generator_optimizer]:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= self.lr_scale

        state.attack_state["lr_step_index"] = step_index + 1
