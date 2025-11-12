"""
Test script to verify ACDNet and Micro-ACDNet implementations
"""

import torch
import numpy as np
from acdnet_model import create_acdnet
from acdnet_micro import create_micro_acdnet


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model, input_shape):
    """Rough FLOP estimation for conv and linear layers"""
    total_flops = 0
    
    def hook(module, input, output):
        nonlocal total_flops
        
        if isinstance(module, torch.nn.Conv1d):
            # FLOPs = (2 * C_in * K - 1) * C_out * L_out
            batch_size = output.shape[0]
            out_channels = output.shape[1]
            output_length = output.shape[2]
            kernel_size = module.kernel_size[0]
            in_channels = module.in_channels
            
            flops = (2 * in_channels * kernel_size - 1) * out_channels * output_length
            total_flops += flops
            
        elif isinstance(module, torch.nn.Conv2d):
            # FLOPs = (2 * C_in * K_h * K_w - 1) * C_out * H_out * W_out
            batch_size = output.shape[0]
            out_channels = output.shape[1]
            output_height = output.shape[2]
            output_width = output.shape[3]
            kernel_size_h = module.kernel_size[0]
            kernel_size_w = module.kernel_size[1]
            in_channels = module.in_channels
            
            flops = (2 * in_channels * kernel_size_h * kernel_size_w - 1) * \
                    out_channels * output_height * output_width
            total_flops += flops
            
        elif isinstance(module, torch.nn.Linear):
            # FLOPs = (2 * in_features - 1) * out_features
            in_features = module.in_features
            out_features = module.out_features
            flops = (2 * in_features - 1) * out_features
            total_flops += flops
    
    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape)
        model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return total_flops


def test_model(model_name, model, input_shape, expected_params=None, expected_size_mb=None):
    """Test a model"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    # Test forward pass
    batch_size = 4
    num_classes = 50
    input_length = 30225
    
    x = torch.randn(batch_size, *input_shape)
    print(f"Input shape: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, num_classes), f"Expected output shape ({batch_size}, {num_classes}), got {output.shape}"
    print("✓ Forward pass successful")
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    if expected_params is not None:
        diff_pct = abs(total_params - expected_params) / expected_params * 100
        if diff_pct < 5:  # Allow 5% difference
            print(f"  ✓ Close to expected: {expected_params:,}")
        else:
            print(f"  ⚠ Expected: {expected_params:,} (diff: {diff_pct:.1f}%)")
    
    # Model size
    size_mb = (total_params * 4) / (1024 ** 2)
    size_mb_int8 = (total_params * 1) / (1024 ** 2)
    print(f"\nModel size:")
    print(f"  FP32: {size_mb:.2f} MB")
    print(f"  INT8: {size_mb_int8:.2f} MB")
    
    if expected_size_mb is not None:
        diff_pct = abs(size_mb - expected_size_mb) / expected_size_mb * 100
        if diff_pct < 5:
            print(f"  ✓ Close to expected: {expected_size_mb:.2f} MB")
        else:
            print(f"  ⚠ Expected: {expected_size_mb:.2f} MB (diff: {diff_pct:.1f}%)")
    
    # Estimate FLOPs
    flops = estimate_flops(model, input_shape)
    flops_millions = flops / 1e6
    print(f"\nEstimated FLOPs: {flops_millions:.2f}M")
    
    # Test different batch sizes
    print("\nTesting different batch sizes:")
    for bs in [1, 2, 8, 16]:
        x = torch.randn(bs, *input_shape)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (bs, num_classes)
        print(f"  ✓ Batch size {bs}: OK")
    
    return {
        'params': total_params,
        'size_mb': size_mb,
        'flops_millions': flops_millions
    }


def compare_models():
    """Compare ACDNet and Micro-ACDNet"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    num_classes = 50
    input_length = 30225
    input_shape = (1, input_length)
    
    # Create models
    acdnet = create_acdnet(num_classes=num_classes, input_length=input_length)
    micro_acdnet = create_micro_acdnet(num_classes=num_classes, input_length=input_length)
    
    # Test ACDNet
    acdnet_stats = test_model(
        "ACDNet",
        acdnet,
        input_shape,
        expected_params=4_740_000,  # 4.74M
        expected_size_mb=18.06
    )
    
    # Test Micro-ACDNet
    micro_stats = test_model(
        "Micro-ACDNet",
        micro_acdnet,
        input_shape,
        expected_params=131_000,  # 0.131M
        expected_size_mb=0.50
    )
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPRESSION STATISTICS")
    print(f"{'='*60}")
    
    size_reduction = (1 - micro_stats['size_mb'] / acdnet_stats['size_mb']) * 100
    param_reduction = (1 - micro_stats['params'] / acdnet_stats['params']) * 100
    flop_reduction = (1 - micro_stats['flops_millions'] / acdnet_stats['flops_millions']) * 100
    
    print(f"Size reduction: {size_reduction:.2f}%")
    print(f"  ACDNet: {acdnet_stats['size_mb']:.2f} MB")
    print(f"  Micro: {micro_stats['size_mb']:.2f} MB")
    print(f"  Target: 97.22%")
    
    print(f"\nParameter reduction: {param_reduction:.2f}%")
    print(f"  ACDNet: {acdnet_stats['params']:,}")
    print(f"  Micro: {micro_stats['params']:,}")
    
    print(f"\nFLOP reduction: {flop_reduction:.2f}%")
    print(f"  ACDNet: {acdnet_stats['flops_millions']:.2f}M")
    print(f"  Micro: {micro_stats['flops_millions']:.2f}M")
    print(f"  Target: 97.28%")
    
    # Check if within expected range
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")
    
    checks = []
    checks.append(("Size reduction ~97%", 95 < size_reduction < 99))
    checks.append(("Micro-ACDNet size ~0.5MB", 0.4 < micro_stats['size_mb'] < 0.6))
    checks.append(("Micro-ACDNet params ~131K", 100_000 < micro_stats['params'] < 150_000))
    
    all_passed = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n{'='*60}")
        print("✓ ALL TESTS PASSED!")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("⚠ SOME TESTS FAILED - Check implementation")
        print(f"{'='*60}")


def test_filter_counts():
    """Test that Micro-ACDNet has the correct filter configuration"""
    print(f"\n{'='*60}")
    print("TESTING FILTER COUNTS")
    print(f"{'='*60}")
    
    model = create_micro_acdnet(num_classes=50)
    filter_counts = model.get_filter_counts()
    
    expected_sfeb = [7, 20]
    expected_tfeb = [10, 14, 22, 31, 35, 41, 51, 67, 69, 48]
    expected_total = sum(expected_sfeb) + sum(expected_tfeb)
    
    print(f"SFEB filters: {filter_counts['sfeb']}")
    print(f"  Expected: {expected_sfeb}")
    print(f"  {'✓' if filter_counts['sfeb'] == expected_sfeb else '✗'} Match")
    
    print(f"\nTFEB filters: {filter_counts['tfeb']}")
    print(f"  Expected: {expected_tfeb}")
    print(f"  {'✓' if filter_counts['tfeb'] == expected_tfeb else '✗'} Match")
    
    print(f"\nTotal filters: {filter_counts['total_filters']}")
    print(f"  Expected: {expected_total}")
    print(f"  {'✓' if filter_counts['total_filters'] == expected_total else '✗'} Match")


if __name__ == "__main__":
    print("ACDNet Implementation Tests")
    print("="*60)
    
    # Test filter counts
    test_filter_counts()
    
    # Compare models
    compare_models()
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)
