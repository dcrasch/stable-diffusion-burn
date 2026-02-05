use std::{error::Error, path::PathBuf, process};

use burn::module::Module;
use burn::record::{self, FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::store::{ApplyResult, BurnpackStore, ModuleSnapshot};
use burn::tensor::backend::Backend;

use indicatif::{ProgressBar, ProgressStyle};
use stablediffusion::{
    model::stablediffusion::{StableDiffusion, StableDiffusionConfig},
    tokenizer::SimpleTokenizer,
};

cfg_if::cfg_if! {
    if #[cfg(feature = "wgpu-backend")] {
        use burn::backend::wgpu::{Wgpu, WgpuDevice};
    }
    else if #[cfg(feature = "rocm-backend")] {
        use burn::backend::rocm::{Rocm, RocmDevice};
    } else if #[cfg(feature = "torch-backend")]{
        use burn::backend::torch::{LibTorch,LibTorchDevice};
    }
    else {
        use burn::backend::ndarray::{NdArray, NdArrayDevice};
    }
}

fn load_stable_diffusion_model_store<B: Backend>(
    filename: &str,
    device: &B::Device,
) -> Result<StableDiffusion<B>, Box<dyn Error>> {
    let model_config = StableDiffusionConfig::new(1000);
    let mut model = model_config.init::<B>(device);

    let tensor_path = PathBuf::from(filename);

    let mut store = BurnpackStore::from_file(tensor_path);
    println!("Loading model");
    let result = model.load_from(&mut store);
    match result {
        Ok(ApplyResult {
            applied,
            skipped,
            missing,
            unused,
            errors,
        }) => {
            println!("missing: {:#?}", missing);
            //println!("unused: {:#?}",unused);
            println!("errors: {:#?}", errors);
            //println!("applied {:#?}", applied);
        }
        Err(e) => {
            return Err(Box::new(e));
        }
    }
    Ok(model)
}

fn load_stable_diffusion_model_file<B: Backend>(
    filename: &str,
    device: &B::Device,
) -> Result<StableDiffusion<B>, record::RecorderError> {
    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
        .load(filename.into(), device)
        .expect("Should decode state successfully");
    Ok(StableDiffusionConfig::new(1000)
        .init(device)
        .load_record(record))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 && args.len() != 8 {
        eprintln!(
            "Usage: {} <model_type(burn or store)> <model_name> <unconditional_guidance_scale> <n_diffusion_steps> <prompt> <output_image_name> [device(cuda, mps, cpu)]",
            args[0]
        );
        process::exit(1);
    }

    let model_type = &args[1];
    let model_name = &args[2];
    let unconditional_guidance_scale: f64 = args[3].parse().unwrap_or_else(|_| {
        eprintln!("Error: Invalid unconditional guidance scale.");
        process::exit(1);
    });
    let n_steps: usize = args[4].parse().unwrap_or_else(|_| {
        eprintln!("Error: Invalid number of diffusion steps.");
        process::exit(1);
    });
    let prompt = &args[5];
    let output_image_name = &args[6];

    cfg_if::cfg_if! {
        if #[cfg(feature = "wgpu-backend")] {
            type Backend = Wgpu;
            let device = WgpuDevice::default();
        } else if #[cfg(feature = "rocm-backend")] {
            type Backend = Rocm;
            let device = RocmDevice::default();
        } else if #[cfg(feature = "torch-backend")] {
            type Backend = LibTorch<f32>;
            // Optional device parameter
            let device_arg = if args.len() == 8 {
                Some(&args[7])
            } else {
                None
            };
            let device = if let Some(dev_str) = device_arg {
                match dev_str.to_lowercase().as_str() {
                    "cpu" => LibTorchDevice::Cpu,
                    "mps" => LibTorchDevice::Mps,
                    s if s.starts_with("cuda") => {
                        let idx = s[4..].parse().unwrap_or(0);
                        LibTorchDevice::Cuda(idx)
                    }
                    _ => {
                        eprintln!("Unknown device: {}", dev_str);
                        process::exit(1);
                    }
                }
            } else {
                LibTorchDevice::Cuda(0)
            };
        }
        else {
            type Backend = NdArray;
            let device = NdArrayDevice::default();
        }
    }

    println!("Loading tokenizer...");
    let tokenizer = SimpleTokenizer::new().unwrap();
    println!("Loading model...");
    let sd: StableDiffusion<Backend> = match model_type.as_str() {
        "burn" => load_stable_diffusion_model_file(model_name, &device).unwrap_or_else(|err| {
            panic!("Error loading model: {}", err);
        }),
        "store" => load_stable_diffusion_model_store(model_name, &device).unwrap_or_else(|err| {
            panic!("Error loading model from store: {}", err);
        }),
        _ => panic!("Unknown model"),
    };

    let unconditional_context = sd.unconditional_context(&tokenizer);
    let context = sd.context(&tokenizer, prompt).unsqueeze::<3>(); //.repeat(0, 2); // generate 2 samples

    println!("Sampling image...");
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}% {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );
    let images = sd.generate_with_callback(
        context,
        unconditional_context,
        unconditional_guidance_scale,
        n_steps,
        |info| {
            let progress = 100 * (info.step + 1) / info.total_steps;
            pb.set_position(progress as u64);
            pb.set_message(format!("Step {}/{}", info.step + 1, info.total_steps));
        },
    );
    save_images(&images, output_image_name, 512, 512).unwrap_or_else(|err| {
        eprintln!("Error saving image: {}", err);
        process::exit(1);
    });
    pb.finish_and_clear();
}

use image::{self, ColorType::Rgb8, ImageResult};

fn save_images(images: &Vec<Vec<u8>>, basepath: &str, width: u32, height: u32) -> ImageResult<()> {
    for (index, img_data) in images.iter().enumerate() {
        let path = format!("{}{}.png", basepath, index);
        image::save_buffer(path, &img_data[..], width, height, Rgb8)?;
    }
    Ok(())
}
