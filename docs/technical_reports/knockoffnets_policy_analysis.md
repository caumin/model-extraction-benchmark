# KnockoffNets Gradient Bandit Policy Analysis

## Current Implementation vs Paper Specifications

### Current Implementation (in `mebench/attackers/knockoff_nets.py`)

The current implementation uses a **simple multi-armed bandit (MAB)** approach with policy gradient updates:

1. **Policy Structure**: Single-level bandit over N classes
   - Policy weights `π` of shape `(num_classes,)`
   - Action selection via `softmax(π)` 
   - Reward: `certainty + diversity + loss` combination

2. **Update Rule** (lines 181-199):
   ```python
   # Simple policy gradient update
   pi = torch.softmax(weights, dim=0)
   for idx, class_id in enumerate(classes):
       alpha = 1.0 / count  # Learning rate decay
       adv = float(rewards[idx]) - baseline
       grad = 1.0 - pi[class_id]  # REINFORCE gradient
       weights = weights + alpha * adv * grad
       pi = torch.softmax(weights, dim=0)
   ```

3. **Hierarchical Elements** (partial):
   - **Coarse clustering**: `coarse_clusters=30` groups classes
   - **Two-level weights**: `policy_weights` (fine) + `coarse_policy_weights` 
   - But only fine-level weights are actually used in updates

### Paper Specifications (Orekondy et al., 2019)

The original KnockoffNets paper proposes a **hierarchical bandit** structure:

1. **Hierarchical Policy**:
   - **Upper level**: Coarse bandit over `K` clusters (e.g., K=30)
   - **Lower level**: For each cluster, fine-grained bandit over member classes
   - **Two-stage selection**: First cluster, then class within cluster

2. **Adaptive Clustering**:
   - Initial clustering using K-means on feature space
   - Clusters updated online as more data is collected
   - Handles the exploration-exploitation tradeoff at two levels

3. **Policy Updates**:
   - Separate policy gradients for upper and lower levels
   - Cluster-level rewards influence class-level learning
   - More sophisticated credit assignment

### Key Differences

| Aspect | Current Implementation | Paper Specification |
|---------|---------------------|---------------------|
| **Policy Structure** | Single-level MAB over N classes | Hierarchical: clusters → classes |
| **Clustering** | Static K-means initialization only | Dynamic, feature-based clustering |
| **Policy Updates** | Direct REINFORCE on class weights | Two-level updates with shared rewards |
| **Exploration** | Count-based decay `α = 1/count` | Upper/lower level exploration strategies |
| **Credit Assignment** | Direct reward → class mapping | Cluster → class reward propagation |

### Impact on Performance

1. **Exploration Efficiency**: 
   - Current: May get stuck in local optima with many classes
   - Paper: Hierarchical exploration reduces effective action space

2. **Sample Complexity**:
   - Current: O(N) exploration for N classes
   - Paper: O(K + N/K) where K << N (typically K=30)

3. **Convergence Speed**:
   - Current: Slower convergence due to large action space
   - Paper: Faster through structured exploration

### Recommendations

1. **Implement True Hierarchical Policy**:
   ```python
   # Two-stage selection
   cluster_probs = softmax(coarse_weights)
   selected_cluster = categorical_sample(cluster_probs)
   
   class_probs = softmax(fine_weights[selected_cluster])
   selected_class = categorical_sample(class_probs)
   ```

2. **Dynamic Clustering**:
   - Update cluster assignments based on collected features
   - Rebalance clusters when class distribution becomes uneven

3. **Proper Credit Assignment**:
   - Upper level: Cluster-level reward aggregation
   - Lower level: Within-cluster competitive updates

4. **Advanced Exploration**:
   - Upper level: Optimistic initialization (UCB-like)
   - Lower level: Thompson sampling or count-based decay

### Current Status

The implementation captures the **core reward computation** (certainty + diversity + loss) correctly, which aligns with the paper. However, the **hierarchical bandit structure** is not fully implemented, which may result in:

- ✅ **Working**: Basic bandit learning with composite rewards
- ❌ **Missing**: Hierarchical exploration and efficiency gains
- ❌ **Missing**: Dynamic cluster adaptation
- ❌ **Missing**: Proper two-level credit assignment

### Files to Modify

- `mebench/attackers/knockoff_nets.py`:
  - `_initialize_policy()` method for hierarchical setup
  - `_select_query_batch()` for two-stage selection
  - `_handle_oracle_output()` for hierarchical updates

This analysis explains the TODO item: "Hierarchical bandit policy 구조 검증" - the current implementation needs verification and completion of the hierarchical structure to match the original paper specifications.