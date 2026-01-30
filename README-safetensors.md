# Stable-Diffusion-Burn with safetensors loading

## Running

```
cargo run --release --bin convert -- sd-v1-4.safetensors sd-1.4.bpk
```

Creates the model with default values, and loads the weights into the model.
It should report, validate and fix paths and data.

The result is a burn pack.
