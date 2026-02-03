
import pytest
import torch
from unittest.mock import MagicMock
from mebench.attackers.blackbox_dissector import BlackboxDissector
from mebench.core.state import BenchmarkState
from mebench.core.types import QueryBatch, OracleOutput

class MockOracle:
    def __init__(self):
        self.model = MagicMock()

class MockContext:
    def __init__(self, state):
        self.state = state
        self.oracle = MockOracle()
        self.budget_remaining = 10000

    def query(self, x, meta=None):
        n = x.size(0)
        self.state.query_count += n
        self.budget_remaining -= n
        # Return dummy hard labels
        y = torch.zeros(n, dtype=torch.long)
        return OracleOutput(y=y, kind="hard_top1")

def test_stage_accounting_exact_split():
    # Setup state
    config = {
        "iterative_budgets": [100],  # First target 100
        "max_budget": 1000,
        "n_variants": 2
    }
    state = BenchmarkState(
        metadata={
            "run_id": "test",
            "seed": 0,
            "dataset_config": {"name": "CIFAR10", "data_mode": "seed"},
            "input_shape": (3, 32, 32),
            "num_classes": 10,
            "max_budget": 1000,
            "device": "cpu"
        }
    )
    
    # Mock dataset creation to avoid loading real data
    # We need to mock create_dataloader or the pool_dataset
    with pytest.MonkeyPatch.context() as m:
        # Create a dummy dataset
        dummy_dataset = [(torch.zeros(3, 32, 32), 0) for _ in range(1000)]
        m.setattr("mebench.attackers.blackbox_dissector.create_dataloader", 
                  lambda *args, **kwargs: MagicMock(dataset=dummy_dataset))
        
        attacker = BlackboxDissector(config, state)
        ctx = MockContext(state)
        attacker.victim = ctx.oracle.model # satisfy run setup
        
        # Manually trigger init if needed (it's in __init__)
        
        # Verify initial targets
        assert state.attack_state["iter_target_q"] == 100
        assert state.attack_state["iter_prev_q"] == 0
        # Delta = 100. A=50, B=50.
        assert state.attack_state["stage_a_remaining"] == 50
        assert state.attack_state["stage_b_remaining"] == 50
        
        # 1. Consume Stage A
        # Request 32 queries
        batch = attacker._select_query_batch(32, state)
        assert batch.meta["stage"] == "A"
        assert batch.x.size(0) == 32
        assert state.attack_state["stage_a_remaining"] == 18 # 50 - 32
        
        # Execute
        ctx.query(batch.x, meta=batch.meta)
        attacker._handle_oracle_output(batch, OracleOutput(y=torch.zeros(32), kind="hard_top1"), state)
        
        # Request 32 queries again (should get 18)
        batch = attacker._select_query_batch(32, state)
        assert batch.meta["stage"] == "A"
        assert batch.x.size(0) == 18
        assert state.attack_state["stage_a_remaining"] == 0
        
        # Execute
        ctx.query(batch.x, meta=batch.meta)
        attacker._handle_oracle_output(batch, OracleOutput(y=torch.zeros(18), kind="hard_top1"), state)
        
        assert state.query_count == 50
        
        # 2. Consume Stage B
        # Now Stage A is empty. Should switch to B.
        # Need to mock substitute or ensure it handles random
        
        batch = attacker._select_query_batch(32, state)
        assert batch.meta["stage"] == "B"
        assert state.attack_state["iter_stage"] == "B"
        assert batch.x.size(0) == 32
        assert state.attack_state["stage_b_remaining"] == 18 # 50 - 32
        
        # Execute
        ctx.query(batch.x, meta=batch.meta)
        attacker._handle_oracle_output(batch, OracleOutput(y=torch.zeros(32), kind="hard_top1"), state)
        
        # Remaining B
        batch = attacker._select_query_batch(32, state)
        assert batch.meta["stage"] == "B"
        assert batch.x.size(0) == 18
        assert state.attack_state["stage_b_remaining"] == 0
        
        # Execute
        ctx.query(batch.x, meta=batch.meta)
        attacker._handle_oracle_output(batch, OracleOutput(y=torch.zeros(18), kind="hard_top1"), state)
        
        assert state.query_count == 100
        
        # 3. Check Transition
        # Now we hit target. _handle_oracle_output should have called _advance
        # Next target should be 1000 (max_budget) since 100 was first
        
        assert state.attack_state["iter_prev_q"] == 100
        assert state.attack_state["iter_target_q"] == 1000
        
        # Delta = 900. A=450, B=450.
        assert state.attack_state["stage_a_remaining"] == 450
        assert state.attack_state["stage_b_remaining"] == 450
        assert state.attack_state["iter_stage"] == "A"

def test_resume_capability():
    # Setup state with existing history
    state = BenchmarkState(
        query_count=100,
        metadata={
            "run_id": "test",
            "seed": 0,
            "dataset_config": {"name": "CIFAR10", "data_mode": "seed"},
            "input_shape": (3, 32, 32),
            "num_classes": 10,
            "max_budget": 1000,
            "device": "cpu"
        }
    )
    # Simulate existing attack state
    state.attack_state = {
        "labeled_indices": [0, 1, 2],
        "unlabeled_indices": [],
        "iter_target_q": 100,
        "iter_prev_q": 0,
        "D_T_x": ["data"], # dummy
        "victim_labels": {0: 0},
        "iter_targets": [100, 200, 1000],
        "iter_ptr": 0
    }
    
    config = {
        "iterative_budgets": [100, 200],
        "max_budget": 1000,
    }
    
    with pytest.MonkeyPatch.context() as m:
        dummy_dataset = [(torch.zeros(3, 32, 32), 0) for _ in range(1000)]
        m.setattr("mebench.attackers.blackbox_dissector.create_dataloader", 
                  lambda *args, **kwargs: MagicMock(dataset=dummy_dataset))
        
        attacker = BlackboxDissector(config, state)
        
        # Verify state was NOT wiped
        assert len(state.attack_state["labeled_indices"]) == 3
        assert state.attack_state["D_T_x"] == ["data"]
        
        # Verify logic proceeds to next target since 100 >= 100
        # attacker.run would loop, but here we check internal logic
        # _advance_iteration_if_needed is called after query
        # But if we just call it manually:
        
        with pytest.MonkeyPatch.context() as m2:
             # Mock train_substitute to avoid logic
             m2.setattr(attacker, "train_substitute", MagicMock())
             attacker._advance_iteration_if_needed(state)
             
        assert state.attack_state["iter_target_q"] == 200


