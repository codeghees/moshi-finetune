# Full Fine-tuning Guide for Moshi

This guide explains how to perform **full fine-tuning** of Moshi models without using LORA (Low-Rank Adaptation). Full fine-tuning trains all model parameters, potentially leading to better performance at the cost of increased computational requirements.

## 🔥 Full Fine-tuning vs LORA

| Aspect | Full Fine-tuning | LORA |
|--------|------------------|------|
| **Parameters trained** | All model parameters (~7B) | Only adapter parameters (~few million) |
| **Memory usage** | Higher | Lower |
| **Training time** | Longer | Shorter |
| **Performance** | Potentially better | Good for most use cases |
| **Checkpoint size** | Full model size (~13GB) | Adapter only (~few MB) |

## 📋 Prerequisites

- At least 80GB GPU memory recommended (for 7B model)
- More storage space for checkpoints
- All standard installation requirements from main README

## ⚙️ Configuration

### Key Configuration Changes

To enable full fine-tuning, you need to modify these settings in your YAML config:

```yaml
# Enable full fine-tuning
full_finetuning: true

# Disable LORA
lora:
  enable: false

# Save full model (not just adapters)
save_adapters: false
```

### Example Configuration

Use the provided `example/moshi_7B_full_finetune.yaml` configuration file:

```yaml
# Full Fine-tuning Configuration for Moshi 7B
full_finetuning: true
lora:
  enable: false
  
# Adjusted hyperparameters for full fine-tuning
batch_size: 8        # Reduced due to memory requirements
optim:
  lr: 1e-6          # Lower learning rate
  
save_adapters: false  # Save full model
```

## 🚀 Training

### Single GPU Training

```bash
torchrun --nproc-per-node 1 -m train example/moshi_7B_full_finetune.yaml
```

### Multi-GPU Training (Recommended)

```bash
torchrun --nproc-per-node 8 --master_port $RANDOM -m train example/moshi_7B_full_finetune.yaml
```

### Memory Requirements

| Setup | Peak Memory per GPU | Recommended GPU |
|-------|-------------------|-----------------|
| 1x GPU | ~75-80GB | H100 80GB |
| 8x GPU | ~25-30GB | A100 40GB or H100 |

## 📊 Expected Performance

With full fine-tuning:

|  | Avg Tokens/sec | Peak Memory | Checkpoint Size |
|------|------|------|------|
| 1×H100 | ~8k | ~75GB | ~13GB |
| 8×H100 | ~7k | ~28GB | ~13GB |

## 🔧 Hyperparameter Recommendations

### Learning Rate
- Start with `1e-6` (lower than LORA)
- Can experiment with `5e-7` to `2e-6`
- Full fine-tuning is more sensitive to learning rate

### Batch Size
- Reduce batch size due to memory constraints
- Recommended: `4-8` per GPU
- Adjust based on available GPU memory

### Steps and Duration
- May need fewer steps than LORA for convergence
- Monitor training loss carefully
- Consider shorter `duration_sec` if memory is tight

## 💾 Checkpointing

### Checkpoint Size
- Full model checkpoints are ~13GB each
- Plan storage accordingly
- Consider reducing `num_ckpt_keep` to save space

### Saving Options
- `save_adapters: false` - Saves complete model
- Checkpoints contain all model weights
- Can be used directly without base model

## 🔮 Inference

### Using Full Fine-tuned Model

After training, use your full fine-tuned model:

```bash
python -m moshi.server \
  --moshi-weight=$CHECKPOINT_DIR/consolidated/consolidated.safetensors \
  --config-path=$CHECKPOINT_DIR/consolidated/config.json
```

### No Base Model Required
- Full fine-tuned checkpoints are standalone
- No need to reference original Moshi weights
- Directly loadable for inference

## 🎯 Best Practices

### 1. Monitor Training Closely
- Full fine-tuning can overfit more easily
- Watch for loss plateaus or increases
- Consider early stopping

### 2. Use Gradient Checkpointing
```yaml
gradient_checkpointing: true
```

### 3. Reduce Batch Size if OOM
- Start with `batch_size: 8`
- Reduce to `4` or `2` if needed
- Adjust `duration_sec` if necessary

### 4. Storage Management
- Each checkpoint is ~13GB
- Use `num_ckpt_keep: 2` to save space
- Regular cleanup of old checkpoints

## 🐛 Troubleshooting

### Out of Memory (OOM)
1. Reduce `batch_size`
2. Reduce `duration_sec`
3. Enable `gradient_checkpointing`
4. Use more GPUs to distribute load

### Slow Training
- Expected with full fine-tuning
- Use multiple GPUs for faster training
- Consider mixed precision training

### Poor Convergence
- Try lower learning rates (`5e-7`)
- Adjust weight decay
- Check data quality and preprocessing

## 📈 Monitoring

### Key Metrics to Watch
- **Training Loss**: Should decrease smoothly
- **Memory Usage**: Monitor for stability
- **Learning Rate**: OneCycleLR schedule
- **Gradient Norm**: Check for gradient explosion

### Weights & Biases Integration
Full fine-tuning produces richer training dynamics:

```yaml
wandb:
  project: "moshi-full-finetune"
  run_name: "full-ft-experiment-1"
```

## 🔄 Converting from LORA to Full Fine-tuning

If you have a LORA configuration, make these changes:

```yaml
# Change these settings
full_finetuning: true    # was: false
lora:
  enable: false          # was: true
save_adapters: false     # was: true

# Adjust these for full fine-tuning
batch_size: 8           # was: 16
optim:
  lr: 1e-6             # was: 2e-6
```

## 🚨 Important Notes

1. **Memory Requirements**: Full fine-tuning requires significantly more GPU memory
2. **Checkpoint Size**: Checkpoints are much larger (~13GB vs ~few MB)
3. **Training Time**: Longer training times compared to LORA
4. **Storage**: Plan for increased storage requirements
5. **Sensitivity**: More sensitive to hyperparameters than LORA

## 💡 When to Use Full Fine-tuning

Consider full fine-tuning when:
- You have sufficient computational resources
- LORA performance is not satisfactory
- You need maximum model customization
- You have a large, high-quality dataset
- You can afford longer training times

Full fine-tuning gives you maximum flexibility and potentially better performance, but requires more resources and careful tuning.