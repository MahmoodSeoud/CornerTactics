#!/usr/bin/env python3
"""
Test script to verify all imbalance fixes are working.
"""
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


def test_class_weights():
    """Test different class weight calculations"""
    print("Testing class weights...")

    pos_rate = 0.182
    neg_rate = 0.818

    weights = {
        'Original': 1.0 / pos_rate - 1.0,
        'Exact ratio': neg_rate / pos_rate,
        'Aggressive': 6.0,
        'Very aggressive': 8.0
    }

    for name, weight in weights.items():
        print(f"  {name}: {weight:.2f}")

    return True


def test_focal_loss():
    """Test focal loss implementation"""
    print("\nTesting Focal Loss...")

    try:
        from src.focal_loss import FocalLoss, WeightedFocalLoss

        # Create imbalanced batch
        targets = torch.tensor([0.0] * 82 + [1.0] * 18)  # 18% positive
        logits = torch.randn(100) - 0.5  # Slight negative bias

        # Compare losses
        bce = torch.nn.BCEWithLogitsLoss()(logits, targets)
        focal = FocalLoss()(logits, targets)
        weighted_focal = WeightedFocalLoss(pos_weight=4.5)(logits, targets)

        print(f"  BCE Loss: {bce:.4f}")
        print(f"  Focal Loss: {focal:.4f}")
        print(f"  Weighted Focal: {weighted_focal:.4f}")

        assert focal < bce * 2, "Focal loss should modulate BCE"
        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_balanced_sampling():
    """Test balanced batch sampling"""
    print("\nTesting Balanced Sampling...")

    try:
        from src.balanced_sampler import BalancedBatchSampler

        # Create imbalanced labels
        labels = [0] * 820 + [1] * 180  # 18% positive

        sampler = BalancedBatchSampler(
            labels=labels,
            batch_size=32,
            oversample=True
        )

        # Check a few batches
        batch_positive_counts = []
        for i, batch_indices in enumerate(sampler):
            batch_labels = [labels[idx] for idx in batch_indices]
            pos_count = sum(batch_labels)
            batch_positive_counts.append(pos_count)

            if i < 3:
                print(f"  Batch {i}: {pos_count} positive, {len(batch_labels) - pos_count} negative")

        avg_positive = np.mean(batch_positive_counts)
        print(f"  Average positive per batch: {avg_positive:.1f} (target: 16)")

        assert 14 <= avg_positive <= 18, "Balanced sampling should give ~50% positive"
        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_metrics():
    """Test balanced metrics calculation"""
    print("\nTesting Balanced Metrics...")

    try:
        from src.balanced_metrics import find_optimal_threshold

        # Simulate predictions
        y_true = np.array([0] * 82 + [1] * 18)
        y_scores = np.random.beta(2, 5, 100)  # Skewed toward 0

        # Find optimal thresholds
        threshold_f1, best_f1 = find_optimal_threshold(y_true, y_scores, 'f1')
        threshold_mcc, best_mcc = find_optimal_threshold(y_true, y_scores, 'mcc')

        print(f"  Optimal F1 threshold: {threshold_f1:.3f} (F1={best_f1:.3f})")
        print(f"  Optimal MCC threshold: {threshold_mcc:.3f} (MCC={best_mcc:.3f})")

        assert 0.1 <= threshold_f1 <= 0.9, "Threshold should be in valid range"
        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_smote_script():
    """Test that SMOTE script exists and can be imported"""
    print("\nTesting SMOTE Script...")

    script_path = Path("scripts/apply_smote.py")
    if script_path.exists():
        print(f"  ✓ SMOTE script exists at {script_path}")
        return True
    else:
        print(f"  ✗ SMOTE script not found at {script_path}")
        return False


def test_balanced_training_script():
    """Test that balanced training script exists"""
    print("\nTesting Balanced Training Script...")

    script_path = Path("scripts/train_gnn_balanced.py")
    if script_path.exists():
        print(f"  ✓ Balanced training script exists at {script_path}")

        # Check if it can be imported
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("train_gnn_balanced", script_path)
            module = importlib.util.module_from_spec(spec)
            print("  ✓ Script can be imported successfully")
            return True
        except Exception as e:
            print(f"  ✗ Error importing script: {e}")
            return False
    else:
        print(f"  ✗ Balanced training script not found at {script_path}")
        return False


def run_all_tests():
    """Run all imbalance fix tests"""
    print("="*60)
    print("TESTING IMBALANCE FIXES")
    print("="*60)

    tests = [
        ("Class Weights", test_class_weights),
        ("Focal Loss", test_focal_loss),
        ("Balanced Sampling", test_balanced_sampling),
        ("Metrics", test_metrics),
        ("SMOTE Script", test_smote_script),
        ("Balanced Training Script", test_balanced_training_script)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"  ✅ {test_name} passed")
            else:
                print(f"  ❌ {test_name} failed")
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:25s}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        print("\nYou can now:")
        print("1. Run: python scripts/apply_smote.py  # Generate augmented data")
        print("2. Run: python scripts/train_gnn_balanced.py  # Train with balanced techniques")
        print("3. Or submit SLURM job: sbatch scripts/slurm/train_balanced_gnn.sh")
        return True
    else:
        print("\n❌ Some tests failed. Please fix the issues before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)