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
use stablediffusion::model::unet::UNetConfig;

fn convert_safetensor_to_model<B: Backend>(
    input_file: &str,
    output_file: &str,
    device: &B::Device,
) -> Result<(), Box<dyn Error>> {
    println!("Loading safetensor...");

    //let clip_config = CLIPConfig::new(49408, 768, 12, 77, 12);
    //let mut model = clip_config.init::<B>(device);

    //let autoencoder_config = AutoencoderConfig::new();
    //let mut model = autoencoder_config.init::<B>(device);
    let unet_config = UNetConfig::new();
    let mut model = unet_config.init::<B>(device);
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
            println!("decoder----");
            println!("missing: {:#?}", missing);
            //println!("unused: {:#?}",unused);
            println!("errors: {:#?}", errors);
            println!("applied {:#?}", applied);
        }
        Err(e) => {
            println!("{:#?}", e);
        }
    }
    /*
    println!("Saving burnpack...");
    let mut store = BurnpackStore::from_file(&output_file)
        .overwrite(true)
        .metadata("format", "safetensor")
        .metadata("description", "Sample file for examining Burnpack format")
        .metadata("version", env!("CARGO_PKG_VERSION"))
        .metadata("author", "Burn Example");
    model.save_into(&mut store).expect("Failed to save model");
    */
    Ok(())
}

fn build_store(path: &Path) -> SafetensorsStore {
    let mut store = SafetensorsStore::from_file(path);
    for &(from, to) in key_remap_rules_unet() {
        store = store.with_key_remapping(from, to);
    }
    store
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true)
        .validate(true)
}
fn key_remap_rules_unet() -> &'static [(&'static str, &'static str)] {
    &[
        (r"model\.diffusion_model\.time_embed.0", "lin1_time_embed"),
        (r"model\.diffusion_model\.time_embed.2", "lin2_time_embed"),
        (r"model\.diffusion_model\.out.0.weight", "norm_out.gamma"),
        (r"model\.diffusion_model\.out.0.bias", "norm_out.beta"),
        (r"model\.diffusion_model\.out.2", "conv_out"),
        // input blocks
        (
            r"model\.diffusion_model\.input_blocks\.0\.0",
            "input_blocks.conv",
        ),
        //(r"model\.diffusion_model\.input_blocks\.1","input_blocks.rt1"),
        //(r"model\.diffusion_model\.input_blocks\.1","input_blocks.rt1"),

        //(r"model\.diffusion_model\.input_blocks\.2","input_blocks.rt2"),
        //(r"model\.diffusion_model\.input_blocks\.3\.0","input_blocks.d1"),
        // resblock 1.0 -> res
        //model.diffusion_model.input_blocks.1.0.in_layers.0.bias
        (
            r"model\.diffusion_model\.input_blocks\.1\.0\.in_layers\.0\.weight",
            "input_blocks.rt1.res.norm_in.gamma",
        ),
        (
            r"model\.diffusion_model\.input_blocks\.1\.0\.in_layers\.0\.bias",
            "input_blocks.rt1.res.norm_in.beta",
        ),
        (
            r"model\.diffusion_model\.input_blocks\.1\.0\.in_layers\.2",
            "input_blocks.rt1.res.conv_in",
        ),
        (
            r"model\.diffusion_model\.input_blocks\.1\.0\.emb_layers\.1\.(.*)",
            "input_blocks.rt1.res.lin_embed.$1",
        ),
        (
            r"model\.diffusion_model\.input_blocks\.1\.0\.out_layers\.0\.weight",
            "input_blocks.rt1.res.norm_out.gamma",
        ),
        (
            r"model\.diffusion_model\.input_blocks\.1\.0\.out_layers\.0\.bias",
            "input_blocks.rt1.res.norm_out.beta",
        ),
        (
            r"model\.diffusion_model\.input_blocks\.1\.0\.out_layers\.3",
            "input_blocks.rt1.res.conv_out",
        ),
        //spatial transformer -> transformer
        (
            r"model\.diffusion_model\.input_blocks\.1\.1\.norm\.weight",
            "input_blocks.rt1.transformer.norm.gamma",
        ),
        (
            r"model\.diffusion_model\.input_blocks\.1\.1\.norm\.bias",
            "input_blocks.rt1.transformer.norm.beta",
        ),
        (
            r"model\.diffusion_model\.input_blocks\.1\.1\.proj_in",
            "input_blocks.rt1.transformer.proj_in",
        ),
        (
            r"model\.diffusion_model\.input_blocks\.1\.1\.proj_out",
            "input_blocks.rt1.transformer.proj_out",
        ),
        (
            r"model\.diffusion_model\.input_blocks\.1\.1\.transformer_blocks\.0\.norm(\d+).weight",
            "input_blocks.rt1.transformer.transformer.norm$1.gamma",
        ),
        (
            r"model\.diffusion_model\.input_blocks\.1\.1\.transformer_blocks\.0\.norm(\d+).bias",
            "input_blocks.rt1.transformer.transformer.norm$1.beta",
        ),
        // cross attention
        (
            r"model\.diffusion_model\.input_blocks\.1\.1\.transformer_blocks\.0\.attn(\d+)\.to_q",
            "input_blocks.rt1.transformer.transformer.attn$1.query",
        ),
        (
            r"model\.diffusion_model\.input_blocks\.1\.1\.transformer_blocks\.0\.attn(\d+)\.to_k",
            "input_blocks.rt1.transformer.transformer.attn$1.key",
        ),
        (
            r"model\.diffusion_model\.input_blocks\.1\.1\.transformer_blocks\.0\.attn(\d+)\.to_v",
            "input_blocks.rt1.transformer.transformer.attn$1.value",
        ),
        (
            r"model\.diffusion_model\.input_blocks\.1\.1\.transformer_blocks\.0\.attn(\d+)\.to_out\.0",
            "input_blocks.rt1.transformer.transformer.attn$1.out",
        ),
        // feed forward
        (
            r"model\.diffusion_model\.input_blocks\.1\.1\.transformer_blocks\.0\.ff\.net.0.proj",
            "input_blocks.rt1.transformer.transformer.mlp.geglu.proj",
        ),
        (
            r"model\.diffusion_model\.input_blocks\.1\.1\.transformer_blocks\.0\.ff\.net.2",
            "input_blocks.rt1.transformer.transformer.mlp.lin",
        ),
    ]
}

fn key_remap_rules_autoencoder() -> &'static [(&'static str, &'static str)] {
    &[
        // autoencoder: first_stage_model
        (r"first_stage_model\.post_quant_conv", "post_quant_conv"),
        (r"first_stage_model\.quant_conv", "quant_conv"),
        // autoencoder: decoder
        (r"first_stage_model\.decoder\.conv_in", "decoder.conv_in"),
        (r"first_stage_model\.decoder\.conv_out", "decoder.conv_out"),
        // autoencoder: decoder.blocks.0.res1.conv2.weight reversed and 1-indexed :-(
        // 0 -> 3
        (
            r"first_stage_model\.decoder\.up\.0\.block\.0\.(.*)",
            "decoder.blocks.3.res1.$1",
        ),
        (
            r"first_stage_model\.decoder\.up\.0\.block\.1\.(.*)",
            "decoder.blocks.3.res2.$1",
        ),
        (
            r"first_stage_model\.decoder\.up\.0\.block\.2\.(.*)",
            "decoder.blocks.3.res3.$1",
        ),
        // 1 -> 2
        (
            r"first_stage_model\.decoder\.up\.1\.block\.0\.(.*)",
            "decoder.blocks.2.res1.$1",
        ),
        (
            r"first_stage_model\.decoder\.up\.1\.block\.1\.(.*)",
            "decoder.blocks.2.res2.$1",
        ),
        (
            r"first_stage_model\.decoder\.up\.1\.block\.2\.(.*)",
            "decoder.blocks.2.res3.$1",
        ),
        (
            r"first_stage_model\.decoder\.up\.1\.upsample\.conv\.(.*)",
            "decoder.blocks.2.upsampler.$1",
        ),
        // 2 -> 1
        (
            r"first_stage_model\.decoder\.up\.2\.block\.0\.(.*)",
            "decoder.blocks.1.res1.$1",
        ),
        (
            r"first_stage_model\.decoder\.up\.2\.block\.1\.(.*)",
            "decoder.blocks.1.res2.$1",
        ),
        (
            r"first_stage_model\.decoder\.up\.2\.block\.2\.(.*)",
            "decoder.blocks.1.res3.$1",
        ),
        (
            r"first_stage_model\.decoder\.up\.2\.upsample\.conv\.(.*)",
            "decoder.blocks.1.upsampler.$1",
        ),
        // 3 -> 0
        (
            r"first_stage_model\.decoder\.up\.3\.block\.0\.(.*)",
            "decoder.blocks.0.res1.$1",
        ),
        (
            r"first_stage_model\.decoder\.up\.3\.block\.1\.(.*)",
            "decoder.blocks.0.res2.$1",
        ),
        (
            r"first_stage_model\.decoder\.up\.3\.block\.2\.(.*)",
            "decoder.blocks.0.res3.$1",
        ),
        (
            r"first_stage_model\.decoder\.up\.3\.upsample\.conv\.(.*)",
            "decoder.blocks.0.upsampler.$1",
        ),
        (
            r"first_stage_model\.decoder\.norm_out.weight",
            "decoder.norm_out.gamma",
        ),
        (
            r"first_stage_model\.decoder\.norm_out.bias",
            "decoder.norm_out.beta",
        ),
        // fix up weights
        (
            r"decoder\.blocks\.(\d+)\.res(\d+).norm(\d+)\.weight",
            "decoder.blocks.$1.res$2.norm$3.gamma",
        ),
        (
            r"decoder\.blocks\.(\d+)\.res(\d+).norm(\d+)\.bias",
            "decoder.blocks.$1.res$2.norm$3.beta",
        ),
        (
            r"decoder\.mid\.block_(\d+)\.norm(\d+)\.bias",
            "decoder.mid.block_$1.norm$2.beta",
        ),
        (
            r"decoder\.mid\.block_(\d+)\.norm(\d+)\.weight",
            "decoder.mid.block_$1.norm$2.gamma",
        ),
        // autoencoder: decoder.mid
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
        // encoder
        (r"first_stage_model\.encoder\.conv_in", "encoder.conv_in"),
        (r"first_stage_model\.encoder\.conv_out", "encoder.conv_out"),
        (
            r"first_stage_model\.encoder\.down\.(\d+)\.block\.0\.(.*)",
            "encoder.blocks.$1.res1.$2",
        ),
        (
            r"first_stage_model\.encoder\.down\.(\d+)\.block\.1\.(.*)",
            "encoder.blocks.$1.res2.$2",
        ),
        (
            r"first_stage_model\.encoder\.down\.(\d+)\.downsample\.conv\.(.*)",
            "encoder.blocks.$1.downsampler.conv.$2",
        ),
        // encoder mid
        (
            r"first_stage_model\.encoder\.mid\.block_(1|2)\.(.*)",
            "encoder.mid.block_$1.$2",
        ),
        (
            r"first_stage_model\.encoder\.mid\.block_(1|2)\.(.*)",
            "encoder.mid.block_$1.$2",
        ),
        (
            r"first_stage_model\.encoder\.mid\.attn_1\.norm\.bias",
            "encoder.mid.attn.norm.beta",
        ),
        (
            r"first_stage_model\.encoder\.mid\.attn_1\.norm\.weight",
            "encoder.mid.attn.norm.gamma",
        ),
        (
            r"first_stage_model\.encoder\.mid\.attn_1\.(q|k|v|proj_out)",
            "encoder.mid.attn.$1",
        ),
        (
            r"first_stage_model\.encoder\.norm_out.weight",
            "encoder.norm_out.gamma",
        ),
        (
            r"first_stage_model\.encoder\.norm_out.bias",
            "encoder.norm_out.beta",
        ),
        // fix up weights
        (
            r"encoder\.blocks\.(\d+)\.res(\d+).norm(\d+)\.weight",
            "encoder.blocks.$1.res$2.norm$3.gamma",
        ),
        (
            r"encoder\.blocks\.(\d+)\.res(\d+).norm(\d+)\.bias",
            "encoder.blocks.$1.res$2.norm$3.beta",
        ),
        (
            r"encoder\.mid\.block_(\d+)\.norm(\d+)\.bias",
            "encoder.mid.block_$1.norm$2.beta",
        ),
        (
            r"encoder\.mid\.block_(\d+)\.norm(\d+)\.weight",
            "encoder.mid.block_$1.norm$2.gamma",
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
