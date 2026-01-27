"""CloudLeak attack implementation."""

from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
from scipy.optimize import fmin_l_bfgs_b

from mebench.attackers.base import BaseAttack
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
        if hasattr(model, "classifier"):
            classifier = model.classifier
            if isinstance(classifier, nn.Sequential):
                # Prefer the last Linear before the final classifier head.
                # VGG: classifier[3] is fc7 (Linear); AlexNet: classifier[4] is fc7 (Linear).
                for idx in [4, 3, 2, 1, 0]:
                    if idx < len(classifier) and isinstance(classifier[idx], nn.Linear):
                        return classifier[idx]
        last_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is None:
            raise ValueError("FeatureFool requires a Linear layer for feature extraction")
        return last_linear

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
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

    def generate(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
    ) -> torch.Tensor:
        """Generate adversarial example from source toward target.

        Args:
            x_source: Source image [C, H, W]
            x_target: Target image [C, H, W]

        Returns:
            Adversarial example [C, H, W]
        """
        self.model.eval()
        with torch.no_grad():
            phi_s = self._extract_features(x_source.unsqueeze(0).to(self.device))
            phi_t = self._extract_features(x_target.unsqueeze(0).to(self.device))

            phi_s_flat = phi_s.view(-1)
            phi_t_flat = phi_t.view(-1)

        # Define objective function for PyTorch LBFGS
        # We optimize delta directly on GPU
        delta = torch.zeros_like(x_source, requires_grad=True, device=self.device)
        optimizer = torch.optim.LBFGS([delta], lr=1.0, max_iter=self.max_iters, history_size=10, line_search_fn="strong_wolfe")
        
        x_source_dev = x_source.to(self.device)
        
        def closure():
            optimizer.zero_grad()
            
            # Adversarial image
            x_adv = torch.clamp(x_source_dev + delta, 0.0, 1.0)
            
            # Extract features (requires separate forward pass logic or hook reset)
            # Since self.model is frozen, we can just run it.
            # But _extract_features registers hooks which might conflict if re-entrant?
            # Safe to call _extract_features inside closure if handle is managed well.
            
            # Optimization: Pre-extract phi_t and phi_s outside closure (already done)
            
            phi_adv = self._extract_features(x_adv.unsqueeze(0)).view(-1)
            
            # Triplet loss
            dist_t = torch.norm(phi_adv - phi_t_flat, p=2)
            dist_s = torch.norm(phi_adv - phi_s_flat, p=2)
            triplet = F.relu(dist_t - dist_s + self.margin_m)
            
            # Visual constraint
            visual_loss = torch.norm(delta, p=2)
            
            loss = visual_loss + self.lambda_adv * triplet
            loss.backward()
            return loss
            
        optimizer.step(closure)
        
        # Final result
        x_adv = torch.clamp(x_source_dev + delta, 0.0, 1.0)
        return x_adv.detach().cpu()


class CloudLeak(BaseAttack):
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
        self.uncertainty_candidates = int(config.get("uncertainty_candidates", 5))

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

        # Pool dataset
        self.pool_dataset = None

        # FeatureFool optimizer
        self.featurefool = None
        self._class_feature_cache: Dict[int, torch.Tensor] = {}

    def _initialize_state(self, state: BenchmarkState) -> None:
        """Initialize attack-specific state.

        Args:
            state: Global benchmark state to update
        """
        # Load dataset config to get size
        dataset_config = state.metadata.get("dataset_config", {})
        if not dataset_config:
            dataset_config = self.config.get("dataset", {})
        
        # We need actual dataset size to random sample
        # Lazy loading or metadata check
        if "size" in dataset_config:
            total_size = dataset_config["size"]
        else:
            # Fallback: load dataset to check size (once)
            temp_loader = create_dataloader(dataset_config, batch_size=1, shuffle=False)
            total_size = len(temp_loader.dataset)

        # Pool tracking: Randomly sample initial pool indices from entire dataset
        # This fixes the bias of taking first N samples
        if total_size > self.initial_pool_size:
            pool_indices = np.random.choice(total_size, self.initial_pool_size, replace=False).tolist()
        else:
            pool_indices = list(range(total_size))
            
        state.attack_state["pool_indices"] = pool_indices

        # Query tracking
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []

        # Synthetic sample tracking
        state.attack_state["synthetic_indices"] = []  # Indices of generated samples

        state.attack_state["substitute"] = None

    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        """Propose k adversarial queries using FeatureFool.

        Args:
            k: Number of queries to propose
            state: Current benchmark state

        Returns:
            QueryBatch with k adversarial queries
        """
        pool_indices = state.attack_state["pool_indices"]

        # Load pool dataset
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

            # Ensure pool_indices matches actual dataset size
            if len(self.pool_dataset) < len(pool_indices):
                print(f"Warning: Dataset size ({len(self.pool_dataset)}) is smaller than initialized pool size. Truncating pool_indices.")
                state.attack_state["pool_indices"] = list(range(len(self.pool_dataset)))
                pool_indices = state.attack_state["pool_indices"]

        # Generate adversarial samples
        substitute = state.attack_state["substitute"]
        device = state.metadata.get("device", "cpu")

        if substitute is None:
            # No substitute yet, use random pool samples
            n_select = min(k, len(pool_indices))
            selected = np.random.choice(pool_indices, n_select, replace=False).tolist()
            x_list = []
            for idx in selected:
                img, _ = self.pool_dataset[idx]
                x_list.append(img)

            x = torch.stack(x_list)
            meta = {"indices": selected, "synthetic": False}
            return QueryBatch(x=x, meta=meta)

        # FeatureFool generation
        x_list = []
        selected_indices = []

        # Initialize FeatureFool if needed
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

        # Generate candidates and select by uncertainty under the substitute.
        n_cand = max(int(k), int(k) * max(1, int(self.uncertainty_candidates)))
        scored = []
        substitute.eval()
        for _ in range(n_cand):
            # Select random source and target from pool
            source_idx = np.random.choice(pool_indices)
            target_idx = np.random.choice([idx for idx in pool_indices if idx != source_idx])

            source_img, source_label = self.pool_dataset[source_idx]
            target_img, _ = self.pool_dataset[target_idx]

            if self.featurefool is not None and source_label is not None:
                margin_m = self._compute_margin_m(int(source_label), device)
                self.featurefool.margin_m = margin_m

            # Generate adversarial example
            x_adv = self.featurefool.generate(source_img, target_img)

            with torch.no_grad():
                probs = F.softmax(substitute(x_adv.unsqueeze(0).to(device)), dim=1).squeeze(0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

            scored.append((entropy, x_adv, (source_idx, target_idx)))

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:k]
        for _entropy, x_adv, pair in top:
            x_list.append(x_adv)
            selected_indices.append(pair)

        x = torch.stack(x_list)
        meta = {
            "indices": selected_indices,
            "synthetic": True,
        }

        return QueryBatch(x=x, meta=meta)

    def _compute_margin_m(self, class_id: int, device: str) -> float:
        if class_id in self._class_feature_cache:
            return float(self._class_feature_cache[class_id])

        if self.pool_dataset is None or self.featurefool is None:
            return float(self.margin_m)

        class_indices = []
        for idx in range(len(self.pool_dataset)):
            _, label = self.pool_dataset[idx]
            if int(label) == class_id:
                class_indices.append(idx)

        if len(class_indices) < 2:
            return float(self.margin_m)

        max_samples = min(32, len(class_indices))
        sampled = np.random.choice(class_indices, max_samples, replace=False).tolist()
        features = []
        for idx in sampled:
            img, _ = self.pool_dataset[idx]
            feats = self.featurefool._extract_features(img.unsqueeze(0).to(device))
            features.append(feats.squeeze(0).detach())

        if len(features) < 2:
            return float(self.margin_m)

        feature_stack = torch.stack(features)
        diffs = feature_stack.unsqueeze(1) - feature_stack.unsqueeze(0)
        distances = torch.norm(diffs, p=2, dim=2)
        mask = ~torch.eye(distances.size(0), dtype=torch.bool, device=distances.device)
        mean_dist = distances[mask].mean().item() if mask.any() else 0.0

        alpha = 0.5
        margin_m = alpha - mean_dist
        self._class_feature_cache[class_id] = torch.tensor(margin_m)
        return float(margin_m)

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        """Observe oracle response and update internal state.

        Args:
            query_batch: The query batch that was sent
            oracle_output: Oracle response
            state: Current benchmark state
        """
        x_batch = query_batch.x
        y_batch = oracle_output.y

        # Store query data
        state.attack_state["query_data_x"].append(x_batch.cpu())
        state.attack_state["query_data_y"].append(y_batch.cpu())

        # Track synthetic samples
        if query_batch.meta.get("synthetic", False):
            indices = query_batch.meta.get("indices", [])
            state.attack_state["synthetic_indices"].extend(indices)

        # Train substitute at the end of each round (paper protocol)
        query_count = len(state.attack_state["query_data_x"])
        if query_count % self.round_size == 0 and query_count > 0:
            self.train_substitute(state)

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
            model = create_substitute(
                arch="resnet18",
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
        model = create_substitute(
            arch="resnet18",
            num_classes=num_classes,
            input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
        ).to(device)

        # Freeze backbone (conv layers)
        # For ResNet18, freeze layer1, layer2, layer3, layer4
        for name, param in model.named_parameters():
            if "fc" not in name:  # Only train final FC
                param.requires_grad = False

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
        for epoch in range(self.max_epochs):
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

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(
                    f"CloudLeak Epoch {epoch}: Loss: {train_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}"
                )

            # Early stopping
            if patience_counter >= self.patience:
                print(f"CloudLeak Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Store in state
        state.attack_state["substitute"] = model

        # Update FeatureFool with new substitute
        self.featurefool = FeatureFool(
            model,
            margin_m=self.margin_m,
            lambda_adv=self.lambda_adv,
            max_iters=self.lbfgs_iters,
            epsilon=self.epsilon,
            device=device,
            config=self.config,
        )

        print(f"CloudLeak substitute trained. Best F1: {best_f1:.4f}")

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
