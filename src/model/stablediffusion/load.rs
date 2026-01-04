use burn::tensor::ElementConversion;
use std::error::Error;

use burn::{
    config::Config,
    module::{Module, Param},
    nn,
    tensor::{Tensor, backend::Backend},
};

use super::*;
use crate::model::{
    autoencoder::load::load_autoencoder, clip::load::load_clip, load::*, unet::load::load_unet,
};

pub fn load_stable_diffusion<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<StableDiffusion<B>, Box<dyn Error>> {
    // stable_diffusion.alphas_cumprod.shape[0]
     let n_steps = load_usize::<B>("n_steps", path, device)?;
    // stable_diffusion.alphas_cumprod
    let alpha_cumulative_products =
        Param::from_tensor(load_tensor::<B, 1>("alphas_cumprod", path, device)?);
    // stable_diffusion.first_stage_model
    let autoencoder = load_autoencoder(&format!("{}/{}", path, "autoencoder"), device)?;

    // stable_diffusion.model.diffusion_model
    let diffusion = load_unet(&format!("{}/{}", path, "unet"), device)?;
    
    // stable_diffusion.cond_stage_model.transformer.text_model
    let clip = load_clip(&format!("{}/{}", path, "clip"), device)?;

    Ok(StableDiffusion {
        n_steps,
        alpha_cumulative_products,
        autoencoder,
        diffusion,
        clip,
    })
}
