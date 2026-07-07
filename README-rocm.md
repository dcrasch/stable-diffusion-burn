# Stable-Diffusion-Burn with ROCM backend

## Running

# rocm (60 seconds)

```
cargo run --release --features rocm-backend --bin sample burn SDv1-4.mpk 7.5 20 "An ancient mossy stone." img
```

Download tensors at: ```https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main```

![An image of an ancient mossy stone](img0.png)

## Convert mode to burn stone

cargo run --release --features rocm-backend --bin convert v1-5-pruned-emaonly.safetensors v1.5.mpk

## Get burn with latest/installed version of rocm

### Get development version from github

```
git clone https://github.com/tracel-ai/burn --branch v0.19.1
git clone https://github.com/tracel-ai/cubecl/ --branch v0.8.1
###--branch   "7.0.5183101"
git clone https://github.com/tracel-ai/cubecl-hip-sys/
## cubek 0.1
git clone https://github.com/tracel-ai/cubek --branch v0.0.1
```

#### Run the exampele

```
cd cubecl
cargo run --example gelu --features hip
```

Error message when ***no HIP_SUCCESS in the root*** this means the hip version is not yet supported by the cubecl_hip_sys crate, it is linking to the wrong hip/rocm version that is installed on the system.

### Build cubecl with latest cubecl-hip-sys

change the cubecl/crates/cubecl-hip/Cargo.toml

```
cubecl-hip-sys = { version = "7.0.5183101" }  to the correct version. ->
cubecl-hip-sys = { path = "../../../cubecl-hip-sys/crates/cubecl-hip-sys", version = "7.1.52802" }
```

```
git clone https://github.com/tracel-ai/cubecl-hip-sys/
cd cubecl-hip-sys
```

Read the readme! And create a version

### Fix cubecl and example

```
cd cubecl
examples/gelu/Cargo.toml
hip = ["cubecl/hip"]
```

examples/gelu/examples/gelu.rs

```
#[cfg(feature = "hip")]    
gelu::launch::<cubecl::hip::HipRuntime>(&Default::default());
```

#### Supported for graphics card (gfx1201)

Is now supported since 0.20.

```
thread 'main' (40614) panicked at crates/cubecl-hip/src/runtime.rs:90:9:
assertion `left == right` failed
````

#### Add arch to amdarchitecture

cubecl-cpp/src/hip/arch.rs
add GFX gfx12, same as gfx11

crates/cubecl-cpp/src/hip/mma/rocwmma_compiler.rs add Architecture::GFX12

### Switch to dev version

comment out the lines ### For local development. ###

* burn/Cargo.toml
* cubek/Cargo.toml
