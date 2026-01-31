#!/bin/bash
"""
Verification test runner for Model Extraction Benchmark compliance.

This script runs all paper compliance tests to validate that attacks
properly implement their respective paper requirements.
"""

echo "🔬 Running Model Extraction Benchmark Compliance Tests..."
echo "=================================================="

# Set environment for reproducibility
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run compliance tests with verbose output
echo "📋 Phase 1: Mathematical Correctness Tests"
pytest tests/verification/test_paper_compliance.py::TestDFMECompliance::test_gradient_estimation_no_dimension_scaling -v
pytest tests/verification/test_paper_compliance.py::TestSwiftThiefCompliance::test_entropy_weighted_cosine_similarity -v
pytest tests/verification/test_paper_compliance.py::TestSwiftThiefCompliance::test_rare_class_switching_condition -v
pytest tests/verification/test_paper_compliance.py::TestESAttackCompliance::test_cross_entropy_loss_not_kl -v

echo "📋 Phase 2: Architecture Compliance Tests"
pytest tests/verification/test_paper_compliance.py::TestArchitectureEnforcement::test_progan_enforcement_blackbox_ripper -v
pytest tests/verification/test_paper_compliance.py::TestArchitectureEnforcement::test_acgan_enforcement_game_es_attack -v
pytest tests/verification/test_paper_compliance.py::TestActiveThiefCompliance::test_dropout_in_substitute_training -v

echo "📋 Phase 3: Hyperparameter Alignment Tests"
pytest tests/verification/test_paper_compliance.py::TestDFMSCompliance::test_initial_query_count_cifar10 -v
pytest tests/verification/test_paper_compliance.py::TestBlackBoxDissectorCompliance::test_max_epochs_200 -v
pytest tests/verification/test_paper_compliance.py::TestCloudLeakCompliance::test_per_class_margin_calculation -v
pytest tests/verification/test_paper_compliance.py::TestGAMECompliance::test_fresh_victim_queries_for_acs -v
pytest tests/verification/test_paper_compliance.py::TestKnockoffNetsCompliance::test_reward_normalization_to_unit_range -v
pytest tests/verification/test_paper_compliance.py::TestCopycatCNNCompliance::test_npdd_dataset_constraint -v

echo "📋 Phase 4: Reproducibility Tests"
pytest tests/verification/test_paper_compliance.py::TestReproducibility::test_fixed_seed_reproducibility -v

echo "🎯 Running Complete Compliance Test Suite..."
pytest tests/verification/test_paper_compliance.py -v --tb=short

echo "✅ Verification tests completed!"
echo "Check test results for paper compliance status."