# vLLM Testing on NERSC Perlmutter

A comprehensive testing framework for vLLM distributed inference on NERSC Perlmutter supercomputer. Test different container backends, models, images, and distributed inference strategies with ease.

## Overview

This repository provides a flexible testing environment for evaluating vLLM performance on Perlmutter's GPU nodes. Whether you're experimenting with different models, container images, or distributed inference methods, this framework simplifies the process with pre-configured setups and automated benchmarking.

## What You Can Test

🐳 **Container Backends**
- **Shifter**: Production-ready container runtime (available now)
- **Podman-HPC**: Next-generation container solution (coming soon)

🤖 **Models**
- Any Hugging Face compatible model
- Pre-configured: Llama 3 8B, Llama 3.3 70B
- Custom model paths and repositories

📦 **Container Images**
- Default: `vllm/vllm-openai:v0.9.1`
- Custom vLLM builds and versions
- Your own optimized images

⚡ **Distributed Inference Methods**
- **Single GPU**: Baseline performance testing
- **Tensor Parallelism (TP)**: Model sharding across GPUs
- **Pipeline Parallelism + Tensor Parallelism (PP+TP)**: Multi-node scaling
- **Data Parallelism**: Multiple model replicas (experimental)

## Quick Start on Perlmutter

### Getting started

```bash
salloc -N 2 -C gpu -t 01:00:00 -q interactive -A <my_account>
# Optional: Set up Hugging Face cache + token
export HF_HOME=$SCRATCH/huggingface
export HF_TOKEN="my-token-here"
```

### Testing

```bash
# Test single GPU performance
make single

# Test tensor parallelism with default 70B model
make tp

# Test multi-node pipeline + tensor parallelism
make pp_tp PP_SIZE=2

# Test with custom model + image
make tp MODEL=microsoft/DialoGPT-large IMAGE=vllm/vllm-openai:v0.8.4
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE` | `single` | Test mode: `single`, `tp`, `pp_tp` |
| `MODEL` | (varies) | Model to test |
| `PP_SIZE` |  | Pipeline parallel size |
| `IMAGE` | `vllm/vllm-openai:v0.9.1` | Container image to test |
| `BACKEND` | `shifter` | Container backend (`podman` coming soon) |
| `HF_HOME` | `$SCRATCH/huggingface/` | Hugging Face cache |

### Test Modes

#### Single GPU Testing (`single`)
- **Purpose**: Baseline performance, small model testing
- **Resources**: 1 GPU, 1 node
- **Default Model**: Llama 3 8B Instruct
- **Use Cases**: Development, model compatibility testing

#### Tensor Parallel Testing (`tp`)
- **Purpose**: Single-node scaling, large model testing
- **Resources**: 4 GPUs, 1 node
- **Default Model**: Llama 3.3 70B Instruct
- **Use Cases**: GPU memory optimization, throughput testing

#### Pipeline + Tensor Parallel Testing (`pp_tp`)
- **Purpose**: Multi-node scaling, very large model testing
- **Resources**: 8 GPUs, 2 nodes (configurable)
- **Default Model**: Llama 3.3 70B Instruct
- **Use Cases**: Extreme scale testing, latency vs throughput analysis

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Shifter User Guide](https://docs.nersc.gov/development/containers/shifter/how-to-use/)

## Support

- **NERSC User Group Slack**: Join the community at [nersc.gov/users/nersc-user-group](https://www.nersc.gov/users/nersc-user-group)