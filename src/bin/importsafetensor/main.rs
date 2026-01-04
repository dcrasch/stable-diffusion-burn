use burn::{
    config::Config,
    module::{Module, Param},
    //nn,
    tensor::{Tensor, backend::Backend},
};
use burn_import::safetensors::{AdapterType, LoadArgs, SafetensorsFileRecorder};

cfg_if::cfg_if! {
    if #[cfg(feature = "wgpu-backend")] {
        use burn::backend::wgpu::{Wgpu, WgpuDevice};
    }
    else if #[cfg(feature = "rocm-backend")] {
       use burn::backend::rocm::{Rocm, RocmDevice};
    } else {
       use burn_tch::{LibTorch, LibTorchDevice};
    }
}

use burn::record::{self, FullPrecisionSettings, Recorder};
use stablediffusion::model::autoencoder::{AutoencoderConfig, Autoencoder};

#[derive(Config, Debug)]
pub struct SDConfig {}
impl SDConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SD<B> {
        let n_steps: usize = 1000;
        let alpha_cumulative_products =
            Param::from_tensor(offset_cosine_schedule_cumprod::<B>(n_steps as i64, device));

        println!("{:?}",alpha_cumulative_products);
        //let autoencoder = AutoencoderConfig::new().init(device);
        //let diffusion = UNetConfig::new().init(device);
        //let clip = CLIPConfig::new(49408, 768, 12, 77, 12).init(device);

        SD {
            n_steps,
            
            alpha_cumulative_products,
            //autoencoder,
            /* 
            diffusion,
            clip,
            */
        }
    }
}

fn offset_cosine_schedule<B: Backend>(n_steps: i64, device: &B::Device) -> Tensor<B, 1> {
    let min_signal_rate: f64 = 0.02;
    let max_signal_rate: f64 = 0.95;
    let start_angle = max_signal_rate.acos();
    let end_angle = min_signal_rate.acos();

    let times = Tensor::arange(1..n_steps + 1, device).float();

    let diffusion_angles = times * ((end_angle - start_angle) / n_steps as f64) + start_angle;
    diffusion_angles.cos()
}
fn offset_cosine_schedule_cumprod<B: Backend>(n_steps: i64, device: &B::Device) -> Tensor<B, 1> {
    offset_cosine_schedule::<B>(n_steps, device).powf_scalar(2.0)
}


#[derive(Module, Debug)]
pub struct SD<B: Backend> {
    n_steps: usize,
    alpha_cumulative_products: Param<Tensor<B, 1>>,
    //autoencoder: Autoencoder<B>,
    /*
    diffusion: UNet<B>,
    clip: CLIP<B>,
    */
}

impl<B: Backend> SD<B> {

}


fn load_stable_diffusion_model_safetensor<B: Backend>(
    filename: &str,
    device: &B::Device,
) -> Result<SD<B>, record::RecorderError>
{
    /*
    NamedMpkFileRecorder::<FullPrecisionSettings>::new()
    .load(filename.into(), device)
        .map(|record| {
            StableDiffusionConfig::new()/
                .init(device)
                .load_record(record)
        })
    */
    //let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
    //    .load(filename.into(), device)
    //    .expect("Should decode state successfully");
/*
    let load_args = LoadArgs::new/(TORCH_WEIGHTS.into())
        // Map *.downsample.0.* -> *.downsample.conv.*
        .with_key_remap("(.+)\\.downsample\\.0\\.(.+)", "$1.downsample.conv.$2")
        // Map *.downsample.1.* -> *.downsample.bn.*
        .with_key_remap("(.+)\\.downsample\\.1\\.(.+)", "$1.downsample.bn.$2")
        // Map layer[i].[j].* -> layer[i].blocks.[j].*
        .with_key_remap("(layer[1-4])\\.([0-9]+)\\.(.+)", "$1.blocks.$2.$3
    */
    /*
    save_scalar(stable_diffusion.alphas_cumprod.shape[0], "n_steps", path)
    save_tensor(stable_diffusion.alphas_cumprod, 'alphas_cumprod', path)
    save_autoencoder(stable_diffusion.first_stage_model, pathlib.Path(path, 'autoencoder'))
    save_unet_model(stable_diffusion.model.diffusion_model, pathlib.Path(path, 'unet'))
    save_clip_text_transformer(stable_diffusion.cond_stage_model.transformer.text_model, pathlib.Path(path, 'clip'))
    */
    // Load weights from Safetensors file
    let load_args = LoadArgs::new(filename.into())
    	.with_key_remap("alphas_cumprod","alpha_cumulative_products")

        // auto encoder
        .with_key_remap("first_stage_model","autoencoder")
        .with_key_remap("attn_1\\.(.+)", "attn.$1")
        //.with_adapter_type(AdapterType::PyTorch) 
        //.with_debug_print(); // Enable debug output
        ;
    let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
        .load(load_args, device)?;

    Ok(SDConfig::new()
        .init(device)
        .load_record(record))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    //let model_type = &args[1];
    let model_name = &args[2];

    cfg_if::cfg_if! {
        if #[cfg(feature = "wgpu-backend")] {
           type Backend = Wgpu;
	   let device = WgpuDevice::default();
    	} else if #[cfg(feature = "rocm-backend")] {
           type Backend = Rocm;
           let device = RocmDevice::default();
        } else {
           type Backend = LibTorch<f32>;
           let device =  LibTorchDevice::Cpu;
        }
    }

    println!("Loading safe tensor model...");
    let sd: SD<Backend> = load_stable_diffusion_model_safetensor::<Backend>(model_name, &device).unwrap();
}