#!/usr/bin/env python3

# YOLOv8n Layer Explorer - Corrected Version
# Properly counts actual parameters, not just tensor count

from ultralytics import YOLO
import torch

# Load model
model = YOLO('yolov8n.pt')

# Get the model structure
print("YOLOv8n Architecture Overview:")
print("=" * 80)

# Method 1: Look at the model.model structure
for i, (name, module) in enumerate(model.model.named_children()):
    print(f"Module {i}: {name}")
    if hasattr(module, '__len__'):
        print(f"  Contains {len(module)} sub-modules")
    print(f"  Type: {type(module).__name__}")
    print()

print("\n" + "=" * 80)
print("Detailed Layer Analysis:")
print("=" * 80)

# Count ACTUAL parameters (not just tensor count)
all_params = list(model.model.named_parameters())
total_params = sum(p.numel() for p in model.model.parameters())
total_tensors = len(all_params)

print(f"Total parameter tensors: {total_tensors}")
print(f"Total parameters: {total_params:,}")
print(f"\nFirst 50 parameter tensors:")

for i, (name, param) in enumerate(all_params[:50]):
    # Extract module number from name
    parts = name.split('.')
    module_num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
    
    param_count = param.numel()
    print(f"{i:3d}: Module {module_num:2d} | {name:60s} | shape={list(param.shape)} | params={param_count:,}")

print("\n" + "=" * 80)
print("YOLOv8 Architecture Breakdown:")
print("=" * 80)

architecture_info = """
YOLOv8n consists of:

BACKBONE (CSPDarknet-based):
- Module 0: Conv (Stem) - Initial convolution
- Module 1: Conv - Downsample
- Module 2: C2f - CSP block with 3 bottlenecks
- Module 3: Conv - Downsample
- Module 4: C2f - CSP block with 6 bottlenecks
- Module 5: Conv - Downsample
- Module 6: C2f - CSP block with 6 bottlenecks
- Module 7: Conv - Downsample
- Module 8: C2f - CSP block with 3 bottlenecks
- Module 9: SPPF - Spatial Pyramid Pooling Fast

NECK (FPN + PAN):
- Module 10: Upsample
- Module 11: Concat
- Module 12: C2f
- Module 13: Upsample
- Module 14: Concat
- Module 15: C2f
- Module 16: Conv
- Module 17: Concat
- Module 18: C2f
- Module 19: Conv
- Module 20: Concat
- Module 21: C2f

HEAD (Detection):
- Module 22: Detect - Final detection layer
"""

print(architecture_info)

# Proper parameter counting by module
print("\n" + "=" * 80)
print("Parameter count by module:")
print("=" * 80)
print("Module | Tensors | Parameters   | Cumulative Params | Cumulative Tensors")
print("-" * 75)

module_param_count = {}
module_tensor_count = {}

# Count parameters and tensors per module
for name, param in all_params:
    parts = name.split('.')
    if len(parts) > 1 and parts[1].isdigit():
        module_num = int(parts[1])
        
        if module_num not in module_tensor_count:
            module_tensor_count[module_num] = 0
            module_param_count[module_num] = 0
        
        module_tensor_count[module_num] += 1
        module_param_count[module_num] += param.numel()

# Display counts with cumulative totals
cum_params = 0
cum_tensors = 0
backbone_params = 0
backbone_tensors = 0

for module_num in sorted(module_param_count.keys()):
    tensors = module_tensor_count[module_num]
    params = module_param_count[module_num]
    cum_tensors += tensors
    cum_params += params
    
    print(f"{module_num:6d} | {tensors:7d} | {params:12,d} | {cum_params:17,d} | {cum_tensors:18d}")
    
    # Mark important boundaries
    if module_num == 9:
        print("-" * 75)
        print(f"BACKBONE END: {cum_tensors} tensors, {cum_params:,d} parameters")
        backbone_tensors = cum_tensors
        backbone_params = cum_params
        print("-" * 75)
    elif module_num == 21:
        print("-" * 75)
        print(f"NECK END: {cum_tensors} tensors, {cum_params:,d} parameters")
        print("-" * 75)

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
print(f"Total model parameters: {total_params:,}")
print(f"Backbone parameters (modules 0-9): {backbone_params:,d} ({backbone_params/total_params*100:.1f}%)")
print(f"Backbone tensor count: {backbone_tensors}")

# Additional analysis
print("\n" + "=" * 80)
print("LAYER TYPE DISTRIBUTION:")
print("=" * 80)

layer_types = {}
for name, module in model.model.named_modules():
    module_type = type(module).__name__
    if module_type not in layer_types:
        layer_types[module_type] = 0
    layer_types[module_type] += 1

for layer_type, count in sorted(layer_types.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{layer_type:20s}: {count:3d} instances")