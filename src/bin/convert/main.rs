use std::error::Error;
use std::path::Path;
use std::process;
use std::{env, path::PathBuf};

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::store::{BurnpackStore, ModuleSnapshot, SafetensorsStore};
use burn::tensor::backend::Backend;

use burn_store::{ApplyResult, PyTorchToBurnAdapter};
use stablediffusion::model::stablediffusion::StableDiffusionConfig;
use stablediffusion::model::clip::CLIPConfig;

fn convert_safetensor_to_model<B: Backend>(
    input_file: &str,
    output_file: &str,
    device: &B::Device,
) -> Result<(), Box<dyn Error>> {
    println!("Loading safetensor...");

    let clip_config = CLIPConfig::new(49408, 768, 12, 77, 12);
    let mut model = clip_config.init::<B>(device);

    let tensor_path = PathBuf::from(input_file);
    let mut store = build_store(&tensor_path);

    println!("Loading model");
    let result = model.load_from(&mut store);
    // TODO report
    // TODO fix stuff
    // TODO validate
    match result  {
        Ok(ApplyResult { applied, skipped,missing, unused, errors,.. }) => { 
            println!("applied {:#?}",applied);
            //println!("missing: {:#?}",missing);
            //println!("unused: {:#?}",unused);
            println!("errors: {:#?}",errors);

    },
        _ => ()
    }
    println!("Saving burnpack...");
    let mut store = BurnpackStore::from_file(&output_file)
        .overwrite(true)
        .metadata("format", "safetensor")
        .metadata("description", "Sample file for examining Burnpack format")
        .metadata("version", env!("CARGO_PKG_VERSION"))
        .metadata("author", "Burn Example");
    model.save_into(&mut store).expect("Failed to save model");
    Ok(())
}

fn build_store(path: &Path) -> SafetensorsStore {
    let mut store = SafetensorsStore::from_file(path);
    for &(from, to) in key_remap_rules() {
        store = store.with_key_remapping(from, to);
    }
    store
     .with_full_path("cond_stage_model.transformer.text_model")
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true)
        .validate(true)
}

fn key_remap_rules() -> &'static [(&'static str, &'static str)] {
    &[
        (r"\.bias$", ".beta"),
        (r"\.weight$", ".gamma"),
        (r"cond_stage_model\.transformer\.text_model\.final_layer_norm\.(.*)","layer_norm.$1"),
    ]
}

fn main() {
    type Backend = NdArray<f32>;
    let device = NdArrayDevice::Cpu;

    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <dump_path> <model_name>", args[0]);
        process::exit(1);
    }

    let dump_path = &args[1];
    let model_name = &args[2];

    if let Err(e) = convert_safetensor_to_model::<Backend>(dump_path, model_name, &device) {
        eprintln!("Failed to convert dump to model: {:?}", e);
        process::exit(1);
    }

    println!("Successfully converted {} to {}", dump_path, model_name);
}
