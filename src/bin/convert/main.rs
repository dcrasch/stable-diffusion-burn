#![allow(dead_code)]
#![allow(unused)]
use std::error::Error;
use std::path::Path;
use std::process;
use std::{env, path::PathBuf};

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::store::{BurnpackStore, ModuleSnapshot, SafetensorsStore};
use burn::tensor::backend::Backend;

use burn_store::{ApplyResult, PyTorchToBurnAdapter};
use stablediffusion::model::autoencoder::AutoencoderConfig;
use stablediffusion::model::clip::CLIPConfig;
use stablediffusion::model::stablediffusion::StableDiffusionConfig;

fn convert_safetensor_to_model<B: Backend>(
    input_file: &str,
    output_file: &str,
    device: &B::Device,
) -> Result<(), Box<dyn Error>> {
    println!("Loading safetensor...");

    //let clip_config = CLIPConfig::new(49408, 768, 12, 77, 12);
    //let mut model = clip_config.init::<B>(device);

    let autoencoder_config = AutoencoderConfig::new();
    let mut model = autoencoder_config.init::<B>(device);
    let tensor_path = PathBuf::from(input_file);
    let mut store = build_store(&tensor_path);

    println!("Loading model");
    let result = model.load_from(&mut store);
    // TODO report
    // TODO fix stuff
    // TODO validate
    match result {
        Ok(ApplyResult {
            applied,
            skipped,
            missing,
            unused,
            errors,
        }) => {
            println!("applied {:#?}", applied);
            println!("missing: {:#?}", missing);
            //println!("unused: {:#?}",unused);
            println!("errors: {:#?}", errors);
        }
        Err(e) => {
            println!("{:#?}", e);
        }
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
    for &(from, to) in key_remap_rules_autoencoder() {
        store = store.with_key_remapping(from, to);
    }
    store
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true)
        .validate(true)
}
fn key_remap_rules_autoencoder() -> &'static [(&'static str, &'static str)] {
    &[
        // autoencoder: first_stage_model
        (r"first_stage_model\.post_quant_conv", "post_quant_conv"),
        (r"first_stage_model\.quant_conv", "quant_conv"),
        (
            r"first_stage_model\.decoder\.mid\.block_(1|2)\.(.*)",
            "decoder.mid.block_$1.$2",
        ),
        (
            r"first_stage_model\.decoder\.mid\.block_(1|2)\.(.*)",
            "decoder.mid.block_$1.$2",
        ),
        (
            r"first_stage_model\.decoder\.mid\.attn_1\.norm\.bias",
            "decoder.mid.attn.norm.beta",
        ),
        (
            r"first_stage_model\.decoder\.mid\.attn_1\.norm\.weight",
            "decoder.mid.attn.norm.gamma",
        ),
        (
            r"first_stage_model\.decoder\.mid\.attn_1\.(q|k|v|proj_out)",
            "decoder.mid.attn.$1",
        ),
    ]
}

fn key_remap_rules_clip() -> &'static [(&'static str, &'static str)] {
    &[
        // clip model: cond_stage_model.transformer.text_model
        (
            r"cond_stage_model\.transformer\.text_model\.final_layer_norm\.bias",
            "layer_norm.beta",
        ),
        (
            r"cond_stage_model\.transformer\.text_model\.final_layer_norm\.weight",
            "layer_norm.gamma",
        ),
        (
            r"cond_stage_model\.transformer\.text_model\.embeddings\.position_embedding\.weight",
            "position_embedding",
        ),
        (
            r"cond_stage_model\.transformer\.text_model\.embeddings\.token_embedding",
            "token_embedding",
        ),
        (
            r"cond_stage_model\.transformer\.text_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj",
            "blocks.$1.attn.key",
        ),
        (
            r"cond_stage_model\.transformer\.text_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj",
            "blocks.$1.attn.out",
        ),
        (
            r"cond_stage_model\.transformer\.text_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj",
            "blocks.$1.attn.query",
        ),
        (
            r"cond_stage_model\.transformer\.text_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj",
            "blocks.$1.attn.value",
        ),
        (
            r"cond_stage_model\.transformer\.text_model\.encoder\.layers\.(\d+)\.layer_norm1\.bias",
            "blocks.$1.attn_ln.beta",
        ),
        (
            r"cond_stage_model\.transformer\.text_model\.encoder\.layers\.(\d+)\.layer_norm1\.weight",
            "blocks.$1.attn_ln.gamma",
        ),
        (
            r"cond_stage_model\.transformer\.text_model\.encoder\.layers\.(\d+)\.mlp\.(.*)",
            "blocks.$1.mlp.$2",
        ),
        (
            r"cond_stage_model\.transformer\.text_model\.encoder\.layers\.(\d+)\.layer_norm2\.bias",
            "blocks.$1.mlp_ln.beta",
        ),
        (
            r"cond_stage_model\.transformer\.text_model\.encoder\.layers\.(\d+)\.layer_norm2\.weight",
            "blocks.$1.mlp_ln.gamma",
        ),
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
