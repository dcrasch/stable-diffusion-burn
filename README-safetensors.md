# Stable-Diffusion-Burn with safetensors loading

## Running conversion
```
cargo run --release --bin convert -- ../../models/sd-v1-4.safetensors sd-v1-4.bpk
```

Creates the model with default values, and loads the weights into the model.
It should report, validate and fix paths and data.

The result is a burn pack.

## running sample
```
cargo run --features wgpu-backend --bin sample store sd-v1-4.bpk 7.5 20 "An ancient mossy stone." img
```

