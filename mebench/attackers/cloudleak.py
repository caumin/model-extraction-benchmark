"""CloudLeak attack implementation."""

from typing import Dict, Any, List, Tuple, Optional
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
from scipy.optimize import fmin_l_bfgs_b
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.models.substitute_factory import create_substitute


class FeatureFool:
    """FeatureFool: Adversarial attack on feature space.

    Generates adversarial examples that maximize feature distance
    while staying visually similar to original.

    Algorithm:
    1. For source image x_s and target image x_t:
       maximize D(phi(x'), phi(x_t)) - D(phi(x'), phi(x_s))
    2. Use L-BFGS for optimization
    3. Enforce visual constraint with pixel-wise L2

    From CloudLeak paper (NDSS 2020).
    """

    def __init__(
        self,
        model: nn.Module,
        margin_m: float = 0.5,
        lambda_adv: float = 0.001,
        max_iters: int = 20,
        epsilon: float = 8.0 / 255.0,
        factr: float = 1e7,
        pgtol: float = 1e-5,
        device: str = "cpu",
        config: Dict[str, Any] | None = None,
    ):
        """Initialize FeatureFool.

        Args:
            model: Substitute model for feature extraction
            margin_m: Triplet loss margin (0.5)
            lambda_adv: Adversarial loss weight (0.001)
            max_iters: L-BFGS max iterations (20)
            epsilon: Max perturbation per pixel (8/255)
            device: Device to use
        """
        self.model = model.eval()
        self.margin_m = margin_m
        self.lambda_adv = lambda_adv
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.factr = factr
        self.pgtol = pgtol
        self.device = device
        self.feature_layer_name = config.get("feature_layer", None) if config else None

        self._feature_layer = self._get_feature_layer(model)

        # Freeze model for adversarial generation
        for param in self.model.parameters():
            param.requires_grad = False

    def _get_feature_layer(self, model: nn.Module) -> nn.Module:
        model_name = model.__class__.__name__.lower()
        if self.feature_layer_name:
            for name, module in model.named_modules():
                if name == self.feature_layer_name:
                    return module
        if "resnet" in model_name and hasattr(model, "avgpool"):
            return model.avgpool
        
        # For LeNet and others with 'classifier' Sequential
        if hasattr(model, "classifier"):
            classifier = model.classifier
            if isinstance(classifier, nn.Sequential):
                # We want the second-to-last Linear layer if possible
                linears = [(i, m) for i, m in enumerate(classifier) if isinstance(m, nn.Linear)]
                if len(linears) >= 2:
                    # Return the one before the last
                    return linears[-2][1]
                if len(linears) == 1:
                    return linears[0][1]
        
        # Fallback to last linear
        last_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is None:
            raise ValueError("FeatureFool requires a Linear layer for feature extraction")
        return last_linear

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using a temporary hook."""
        activations: List[torch.Tensor] = []

        def forward_hook(_module, _inputs, output):
            activations.append(output)

        handle = self._feature_layer.register_forward_hook(forward_hook)
        _ = self.model(x)
        handle.remove()

        if not activations:
            raise RuntimeError("Feature layer hook failed to capture features")
        feats = activations[0]
        if feats.ndim > 2:
            feats = feats.view(feats.size(0), -1)
        return feats

    def generate_batch(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        margin_m: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate a batch of adversarial examples.

        Args:
            x_source: Source images [B, C, H, W]
            x_target: Target images [B, C, H, W]
            margin_m: Per-sample margins [B]. If None, uses self.margin_m.

        Returns:
            Adversarial examples [B, C, H, W]
        """
        B = x_source.size(0)
        self.model.eval()
        
        if margin_m is None:
            margin_m = torch.full((B,), self.margin_m, device=self.device)
        else:
            margin_m = margin_m.to(self.device)
        
        # Setup persistent hook for the duration of optimization
        activations: List[torch.Tensor] = []
        def forward_hook(_module, _inputs, output):
            activations.append(output)
        
        hook_handle = self._feature_layer.register_forward_hook(forward_hook)
        
        try:
            x_source_dev = x_source.to(self.device)
            with torch.no_grad():
                # Initial feature extraction
                _ = self.model(x_source_dev)
                phi_s = activations.pop(0).detach().view(B, -1)
                
                _ = self.model(x_target.to(self.device))
                phi_t = activations.pop(0).detach().view(B, -1)
                
            # Optimize delta [B, C, H, W] with epsilon constraint
            delta = torch.zeros_like(x_source, requires_grad=True, device=self.device)
            
            # Use L-BFGS
            optimizer = torch.optim.LBFGS([delta], lr=1.0, max_iter=self.max_iters, history_size=10, line_search_fn="strong_wolfe")
            
            margin_m = margin_m.view(B, 1)
            
            def closure():
                optimizer.zero_grad()
                # Enforce epsilon constraint and box constraint
                delta_clamped = torch.clamp(delta, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x_source_dev + delta_clamped, 0.0, 1.0)
                
                # Forward pass to trigger hook
                _ = self.model(x_adv)
                phi_adv = activations.pop(0).view(B, -1)
                
                # Per-sample triplet loss
                dist_t = torch.norm(phi_adv - phi_t, p=2, dim=1).view(B, 1)
                dist_s = torch.norm(phi_adv - phi_s, p=2, dim=1).view(B, 1)
                
                triplet = torch.clamp(dist_t - dist_s + margin_m, min=0.0)
                
                # Visual loss: L2 norm squared of delta (as per paper)
                visual_loss = torch.sum(delta_clamped ** 2, dim=(1, 2, 3)).view(B, 1)
                
                loss = torch.mean(visual_loss + self.lambda_adv * triplet)
                loss.backward()
                return loss
                
            optimizer.step(closure)
            
            with torch.no_grad():
                delta_final = torch.clamp(delta, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x_source_dev + delta_final, 0.0, 1.0)
            return x_adv.detach().cpu()
        finally:
            hook_handle.remove()

    def generate(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback for single image generation."""
        margin = torch.tensor([self.margin_m])
        return self.generate_batch(x_source.unsqueeze(0), x_target.unsqueeze(0), margin).squeeze(0)


class CloudLeak(AttackRunner):
    """CloudLeak: Adversarial active learning for model extraction.

    Algorithm loop:
    1. Start with random seed pool X0
    2. Generate adversarial samples using FeatureFool:
       - Select source and target from pool
       - Optimize toward maximizing feature distance
    3. Query victim on adversarial samples
    4. Select samples with highest uncertainty (least confidence)
    5. Train substitute via transfer learning:
       - Freeze pretrained backbone
       - Train final FC only
    6. Repeat until budget exhausted

    Hyperparameters (from CloudLeak repo + paper):
    - lbfgs_iters: L-BFGS max iterations (20)
    - lbfgs_factr: L-BFGS factr parameter (10000000.0)
    - lbfgs_pgtol: L-BFGS pgtol parameter (1e-05)
    - margin_m: Triplet loss margin (0.5)
    - lambda_adv: Adversarial loss weight (smallest value, 0.001)
    - initial_pool_size: Initial seed pool size (1000)
    - batch_size: Training batch size (64)
    - lr: Learning rate (0.01, SGD from Caffe)
    """

    def __init__(self, config: dict, state: BenchmarkState):
        """Initialize CloudLeak attack.

        Args:
            config: Attack configuration
            state: Global benchmark state
        """
        super().__init__(config, state)

        # L-BFGS hyperparameters
        self.lbfgs_iters = int(config.get("lbfgs_iters", 20))
        self.lbfgs_factr = float(config.get("lbfgs_factr", 10000000.0))
        self.lbfgs_pgtol = float(config.get("lbfgs_pgtol", 1e-05))

        # Adversarial synthesis hyperparameters
        self.margin_m = float(config.get("margin_m", 0.5))
        self.lambda_adv = float(config.get("lambda_adv", 0.001))
        self.epsilon = float(config.get("epsilon", 8.0 / 255.0))

        # Round-based hyperparameters (paper ~1000 per round)
        self.num_rounds = int(config.get("num_rounds", 10))
        total_budget = int(state.metadata.get("max_budget", 10000))
        self.round_size = max(1, total_budget // self.num_rounds)
        
        # Missing attribute restored (Paper implies ~20% or min samples per class)
        # We align with ActiveThief benchmark standard: 10% of total budget
        self.initial_pool_size = int(config.get("initial_pool_size", max(1, int(0.1 * total_budget))))
        
        # Training hyperparameters
        self.batch_size = int(config.get("batch_size", 64))
        self.lr = float(config.get("lr", 0.01))
        self.momentum = float(config.get("momentum", 0.9))
        self.weight_decay = float(config.get("weight_decay", 5e-4))
        self.max_epochs = int(config.get("max_epochs", 1000))
        self.patience = int(config.get("patience", 100))

        # Initialize attack state
        self._initialize_state(state)
        
        self.pool_dataset = None
        self.featurefool = None
        self._class_feature_cache: Dict[int, torch.Tensor] = {}
        self._class_indices_cache: Dict[int, List[int]] = {}

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        device = self.state.metadata.get("device", "cpu")
        total_budget = self.state.budget_remaining
        pbar = tqdm(total=total_budget, desc="[CloudLeak] Extracting")
        
        while ctx.budget_remaining > 0:
            step_size = self._default_step_size(ctx)
            x_query, meta = self._select_query_batch(step_size, self.state)
            oracle_output = ctx.query(x_query, meta=meta)
            self._handle_oracle_output(x_query, meta, oracle_output, self.state)
            pbar.update(x_query.size(0))
        pbar.close()
        # Final Evaluation (handled by engine)

        # Pool dataset
        self.pool_dataset = None

        # FeatureFool optimizer
        self.featurefool = None
        self._class_feature_cache: Dict[int, torch.Tensor] = {}
        self._class_indices_cache: Dict[int, List[int]] = {}

    def _select_query_batch(self, k: int, state: BenchmarkState) -> tuple[torch.Tensor, dict]:
        pool_indices = state.attack_state["pool_indices"]

        if self.pool_dataset is None:
            dataset_config = state.metadata.get("dataset_config", {})
            if "data_mode" not in dataset_config:
                dataset_config = {"data_mode": "seed", **dataset_config}
            if "name" not in dataset_config:
                dataset_config = {"name": "CIFAR10", **dataset_config}
            self.pool_dataset = create_dataloader(
                dataset_config,
                batch_size=1,
                shuffle=False,
            ).dataset

            if len(self.pool_dataset) < len(pool_indices):
                self.logger.warning(
                    "Dataset size (" + str(len(self.pool_dataset)) + ") is smaller than initialized pool size. Truncating pool_indices."
                )
                state.attack_state["pool_indices"] = list(range(len(self.pool_dataset)))
                pool_indices = state.attack_state["pool_indices"]

        substitute = state.attack_state["substitute"]
        device = state.metadata.get("device", "cpu")

        if substitute is None:
            n_select = min(k, len(pool_indices))
            selected = np.random.choice(pool_indices, n_select, replace=False).tolist()
            x_list = []
            for idx in selected:
                img, _ = self.pool_dataset[idx]
                x_list.append(img)

            x = torch.stack(x_list)
            meta = {"indices": selected, "synthetic": False}
            return x, meta

        x_list = []
        selected_indices = []
        scored = []

        if self.featurefool is None:
            self.featurefool = FeatureFool(
                substitute,
                margin_m=self.margin_m,
                lambda_adv=self.lambda_adv,
                max_iters=self.lbfgs_iters,
                epsilon=self.epsilon,
                factr=self.lbfgs_factr,
                pgtol=self.lbfgs_pgtol,
                device=device,
                config=self.config,
            )

        substitute.eval()
        
        # Use the ENTIRE pool for adversarial query generation (Strict Protocol)
        all_indices = pool_indices
        n_total = len(all_indices)

        # print(
        #     "\nGenerating adversarial queries for the ENTIRE pool (" + str(n_total) + " samples) via FeatureFool..."
        # )

        # Prepare label lookup and dissimilar indices for O(N) target selection
        label_to_indices = {}
        idx_to_label = {}
        for idx in all_indices:
            _, label = self.pool_dataset[idx]
            l = int(label)
            idx_to_label[idx] = l
            if l not in label_to_indices:
                label_to_indices[l] = []
            label_to_indices[l].append(idx)
        
        all_labels_set = set(label_to_indices.keys())

        batch_size = int(self.config.get("attack", {}).get("gen_batch_size", 64))
        gen_pbar = tqdm(range(0, n_total, batch_size), desc="[CloudLeak] Generating Adversarial Queries", leave=False)
        for i in gen_pbar:
            end_idx = min(i + batch_size, n_total)
            curr_indices = all_indices[i:end_idx]

            s_imgs = []
            t_imgs = []
            for s_idx in curr_indices:
                s_img, _ = self.pool_dataset[s_idx]
                s_imgs.append(s_img)

                # O(1) target selection
                source_label = idx_to_label[s_idx]
                other_labels = list(all_labels_set - {source_label})
                
                if other_labels:
                    target_label = np.random.choice(other_labels)
                    target_idx = np.random.choice(label_to_indices[target_label])
                else:
                    # Fallback to random if only one class (unlikely)
                    other_indices = [idx for idx in all_indices if idx != s_idx]
                    target_idx = np.random.choice(other_indices) if other_indices else s_idx
                
                t_img, _ = self.pool_dataset[target_idx]
                t_imgs.append(t_img)

            s_imgs = torch.stack(s_imgs)
            t_imgs = torch.stack(t_imgs)

            s_imgs = s_imgs.to(device)
            t_imgs = t_imgs.to(device)

            # [P0 FIX] Use per-class margin M instead of None
            margin_m = self._compute_margin_m(curr_indices[j], device) if len(curr_indices) > 1 else None
            s_imgs_adv = self.featurefool.generate_batch(s_imgs, t_imgs, margin_m=margin_m)

            with torch.no_grad():
                logits = substitute(s_imgs_adv.to(device))
                probs = F.softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

            for j in range(s_imgs_adv.size(0)):
                scored.append((float(entropy[j].item()), curr_indices[j], s_imgs_adv[j].detach().cpu()))

        scored.sort(key=lambda t: t[0], reverse=True)
        top_scored = scored[:k]

        for score, idx, adv_img in top_scored:
            selected_indices.append(int(idx))
            x_list.append(adv_img)

        if len(x_list) == 0:
            raise ValueError("CloudLeak selection produced empty batch.")

        x = torch.stack(x_list)
        meta = {"indices": selected_indices, "synthetic": True}
        return x, meta

    def _handle_oracle_output(
        self,
        x_batch: torch.Tensor,
        meta: dict,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        y_batch = oracle_output.y

        state.attack_state["query_data_x"].append(x_batch.cpu())
        state.attack_state["query_data_y"].append(y_batch.cpu())

        if meta.get("synthetic", False):
            indices = meta.get("indices", [])
            state.attack_state["synthetic_indices"].extend(indices)

        query_count = sum(len(x) for x in state.attack_state["query_data_x"])
        if query_count % self.round_size == 0 and query_count > 0:
            self.train_substitute(state)

    def _initialize_state(self, state: BenchmarkState) -> None:
        """Initialize attack-specific state.

        Args:
            state: Global benchmark state to update
        """
        # Load dataset config to get size
        dataset_config = state.metadata.get("dataset_config", {})
        if not dataset_config:
            dataset_config = self.config.get("dataset", {})
        
        # We need actual dataset size to initialize pool
        if self.pool_dataset is None:
            self.pool_dataset = create_dataloader(
                dataset_config,
                batch_size=1,
                shuffle=False,
            ).dataset
        
        total_size = len(self.pool_dataset)
        
        # CloudLeak starts with a full unlabeled pool and no initial labeled set
        # unless specified. We treat the entire dataset as the pool.
        state.attack_state["pool_indices"] = list(range(total_size))

        # Query tracking
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []

        # Synthetic sample tracking
        state.attack_state["synthetic_indices"] = []  # Indices of generated samples

        state.attack_state["substitute"] = None

    def _compute_margin_m(self, class_id: int, device: str) -> float:
        if class_id in self._class_feature_cache:
            return float(self._class_feature_cache[class_id])

        if self.pool_dataset is None or self.featurefool is None:
            return float(self.margin_m)

        # Index dataset by class if not already done
        if not self._class_indices_cache:
            for idx in range(len(self.pool_dataset)):
                _, label = self.pool_dataset[idx]
                l = int(label)
                if l not in self._class_indices_cache:
                    self._class_indices_cache[l] = []
                self._class_indices_cache[l].append(idx)
            # print(f"Indexed pool dataset ({len(self.pool_dataset)} samples) for margin computation.")

        class_indices = self._class_indices_cache.get(class_id, [])

        if len(class_indices) < 2:
            return float(self.margin_m)

        # Use ALL samples in the class to compute average squared distance (Strict Protocol)
        sampled = class_indices
        features = []
        for idx in sampled:
            img, _ = self.pool_dataset[idx]
            feats = self.featurefool._extract_features(img.unsqueeze(0).to(device))
            features.append(feats.squeeze(0).detach())

        if len(features) < 2:
            return float(self.margin_m)

        feature_stack = torch.stack(features)
        # diffs: [N, N, D]
        diffs = feature_stack.unsqueeze(1) - feature_stack.unsqueeze(0)
        # squared distances: [N, N]
        sq_distances = torch.sum(diffs ** 2, dim=2)
        mask = ~torch.eye(sq_distances.size(0), dtype=torch.bool, device=sq_distances.device)
        
        # Paper Eq. 11: average squared distance
        mean_sq_dist = sq_distances[mask].mean().item() if mask.any() else 0.0

        alpha = 0.5
        margin_m = alpha - mean_sq_dist
        self._class_feature_cache[class_id] = torch.tensor(margin_m)
        return float(margin_m)

    def train_substitute(self, state: BenchmarkState) -> None:
        """Train substitute model via transfer learning.

        Freeze pretrained backbone, train final FC only.

        Args:
            state: Current benchmark state
        """
        query_data_x = state.attack_state["query_data_x"]
        query_data_y = state.attack_state["query_data_y"]

        if len(query_data_x) == 0:
            return

        # Concatenate query data
        x_all = torch.cat(query_data_x, dim=0)
        y_all = torch.cat(query_data_y, dim=0)

        # Create dataset
        class QueryDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        dataset = QueryDataset(x_all, y_all)

        # Use 20% validation split
        total_size = len(dataset)
        val_size = max(1, int(0.2 * total_size))
        train_size = total_size - val_size

        if train_size < 2:
            device = state.metadata.get("device", "cpu")
            num_classes = int(
                state.metadata.get("num_classes")
                or state.metadata.get("dataset_config", {}).get("num_classes", 10)
            )
            sub_config = state.metadata.get("substitute_config", {})
            model = create_substitute(
                arch=sub_config.get("arch", "resnet18"),
                num_classes=num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
            ).to(device)
            state.attack_state["substitute"] = model
            return

        train_subset, val_subset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Initialize model with pretrained backbone
        device = state.metadata.get("device", "cpu")
        num_classes = int(
            state.metadata.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        sub_config = state.metadata.get("substitute_config", {})
        model = create_substitute(
            arch=sub_config.get("arch", "resnet18"),
            num_classes=num_classes,
            input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
        ).to(device)

        # Freeze backbone (conv layers)
        # We find the final Linear layers (head) and keep them trainable
        # Common names: 'fc', 'classifier', 'last_linear'
        head_keywords = ["fc", "classifier", "last_linear"]
        for name, param in model.named_parameters():
            if not any(k in name for k in head_keywords):
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Verify if any parameters are trainable
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            self.logger.warning("No trainable parameters found with current head keywords. Unfreezing last parameter.")
            # Fallback: unfreeze the last layer's parameters
            all_params = list(model.parameters())
            if all_params:
                all_params[-1].requires_grad = True
                all_params[-2].requires_grad = True if len(all_params) > 1 else False

        # Optimizer (only for FC layer)
        import torch.optim as optim
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Early stopping
        best_f1 = 0.0
        patience_counter = 0
        best_model_state = None

        # Training loop
        epoch_pbar = tqdm(range(self.max_epochs), desc="[CloudLeak] Training Substitute", leave=False)
        for epoch in epoch_pbar:
            model.train()
            train_loss = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(x_batch)

                # Handle hard labels
                y_labels = y_batch.argmax(dim=1) if y_batch.ndim > 1 else y_batch

                loss = criterion(outputs, y_labels.long())

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            val_f1 = self._compute_f1(model, val_loader, device)
            epoch_pbar.set_postfix({"Loss": f"{train_loss/len(train_loader):.4f}", "F1": f"{val_f1:.4f}"})

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            # if epoch % 10 == 0:
            #     print(
            #         f"CloudLeak Epoch {epoch}: Loss: {train_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}"
            #     )

            # Early stopping
            if patience_counter >= self.patience:
                self.logger.info(f"CloudLeak Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Store in state
        state.attack_state["substitute"] = model
        self.logger.info(f"CloudLeak substitute trained. Best Val F1: {best_f1:.4f}")
        self._evaluate_current_substitute(model, device)

        # Update FeatureFool with new substitute and clear margin cache
        self._class_feature_cache = {}
        self.featurefool = FeatureFool(
            model,
            margin_m=self.margin_m,
            lambda_adv=self.lambda_adv,
            max_iters=self.lbfgs_iters,
            epsilon=self.epsilon,
            device=device,
            config=self.config,
        )

    def _compute_f1(self, model: nn.Module, val_loader: DataLoader, device: str) -> float:
        """Compute F1 score on validation set.

        Args:
            model: Model to evaluate
            val_loader: Validation data loader
            device: Device to use

        Returns:
            F1 score (macro average)
        """
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                outputs = model(x_batch)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)

                if y_batch.ndim > 1:
                    targets = torch.argmax(y_batch, dim=1).cpu().numpy()
                else:
                    targets = y_batch.cpu().numpy()
                all_targets.extend(targets)

        return f1_score(all_targets, all_preds, average="macro")
