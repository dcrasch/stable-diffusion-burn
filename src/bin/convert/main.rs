#![allow(dead_code)]
#![allow(unused)]
use std::error::Error;
use std::path::Path;
use std::process;
use std::{env, path::PathBuf};

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::store::{BurnpackStore, ModuleSnapshot, SafetensorsStore};
use burn::tensor::backend::Backend;

use burn::store::{ApplyResult, KeyRemapper, PyTorchToBurnAdapter};
use stablediffusion::model::stablediffusion::{StableDiffusion, StableDiffusionConfig};

fn load_pretrained<B: Backend>(
    checkpoint: &str,
    device: &B::Device,
) -> Result<StableDiffusion<B>, Box<dyn Error>> {
    println!("Loading safetensor...");
    let key_mappings = key_remap_rules_sd();
    let remapper = KeyRemapper::from_patterns(key_mappings).expect("Invalid key mapping regex");
    let mut store = SafetensorsStore::from_file(checkpoint)
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(remapper)
        .allow_partial(true)
        .validate(true);
    let sd_config = StableDiffusionConfig::default();
    let mut model = sd_config.init::<B>(device);
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

fn key_remap_rules_sd() -> Vec<(String, String)> {
    let mut rules: Vec<(String, String)> = Vec::new();
    key_remap_rules_clip()
        .iter()
        .for_each(|(a, b)| rules.push((format!("{}", a), format!("clip.{}", b))));
    key_remap_rules_autoencoder()
        .iter()
        .for_each(|(a, b)| rules.push((format!("{}", a), format!("autoencoder.{}", b))));
    key_remap_rules_unet()
        .iter()
        .for_each(|(a, b)| rules.push((format!("{}", a), format!("{}", b))));
    rules.push((
        format!("alphas_cumprod"),
        format!("alpha_cumulative_products"),
    ));
    rules
}
fn key_remap_rules_unet() -> &'static [(&'static str, &'static str)] {
    &[
        (
            r"model\.diffusion_model\.time_embed\.0",
            "diffusion.lin1_time_embed",
        ),
        (
            r"model\.diffusion_model\.time_embed\.2",
            "diffusion.lin2_time_embed",
        ),
        (r"model\.diffusion_model\.out\.0", "diffusion.norm_out"),
        (r"model\.diffusion_model\.out\.2", "diffusion.conv_out"),
        (
            r"model\.diffusion_model\.input_blocks",
            "diffusion.input_blocks",
        ),
        (
            r"model\.diffusion_model\.output_blocks",
            "diffusion.output_blocks",
        ),
        (
            r"model\.diffusion_model\.middle_block",
            "diffusion.middle_block",
        ),
        (r"input_blocks\.0\.0", "input_blocks.conv"),
        (r"input_blocks\.1\.", "input_blocks.rt1."),
        (r"input_blocks\.2\.", "input_blocks.rt2."),
        (r"input_blocks\.3\.0.op", "input_blocks.d1"),
        (r"input_blocks\.4\.", "input_blocks.rt3."),
        (r"input_blocks\.5\.", "input_blocks.rt4."),
        (r"input_blocks\.6\.0.op", "input_blocks.d2"),
        (r"input_blocks\.7\.", "input_blocks.rt5."),
        (r"input_blocks\.8\.", "input_blocks.rt6."),
        (r"input_blocks\.9\.0.op", "input_blocks.d3"),
        (r"input_blocks\.10\.", "input_blocks.r1."),
        (r"input_blocks\.11\.", "input_blocks.r2."),
        (r"output_blocks\.0\.", "output_blocks.r1."),
        (r"output_blocks\.1\.", "output_blocks.r2."),
        (r"output_blocks\.2\.", "output_blocks.ru."),
        (r"output_blocks\.3\.", "output_blocks.rt1."),
        (r"output_blocks\.4\.", "output_blocks.rt2."),
        (r"output_blocks\.5\.", "output_blocks.rtu1."),
        (r"output_blocks\.6\.", "output_blocks.rt3."),
        (r"output_blocks\.7\.", "output_blocks.rt4."),
        (r"output_blocks\.8\.", "output_blocks.rtu2."),
        (r"output_blocks\.9\.", "output_blocks.rt5."),
        (r"output_blocks\.10\.", "output_blocks.rt6."),
        (r"output_blocks\.11\.", "output_blocks.rt7."),
        (r"(rt\d+|ru|rtu\d+)\.0\.in_layers\.0", "$1.res.norm_in"),
        (r"(rt\d+|ru|rtu\d+)\.0\.in_layers\.2", "$1.res.conv_in"),
        (r"(rt\d+|ru|rtu\d+)\.0\.emb_layers\.1", "$1.res.lin_embed"),
        (r"(rt\d+|ru|rtu\d+)\.0\.out_layers\.0", "$1.res.norm_out"),
        (r"(rt\d+|ru|rtu\d+)\.0\.out_layers\.3", "$1.res.conv_out"),
        (
            r"(rt\d+|ru|rtu\d+)\.0\.skip_connection",
            "$1.res.skip_connection",
        ),
        (r"(ru)\.1", "$1.upsample"),
        (r"(rtu\d+)\.2", "$1.upsample"),
        (r"(r\d+)\.0\.in_layers\.0", "$1.norm_in"),
        (r"(r\d+)\.0\.in_layers\.2", "$1.conv_in"),
        (r"(r\d+)\.0\.emb_layers\.1", "$1.lin_embed"),
        (r"(r\d+)\.0\.out_layers\.0", "$1.norm_out"),
        (r"(r\d+)\.0\.out_layers\.3", "$1.conv_out"),
        (r"(r\d+)\.0\.skip_connection", "$1.skip_connection"),
        (
            r"middle_block\.0\.in_layers\.0",
            "middle_block.res1.norm_in",
        ),
        (
            r"middle_block\.0\.in_layers\.2",
            "middle_block.res1.conv_in",
        ),
        (
            r"middle_block\.0\.emb_layers\.1",
            "middle_block.res1.lin_embed",
        ),
        (
            r"middle_block\.0\.out_layers\.0",
            "middle_block.res1.norm_out",
        ),
        (
            r"middle_block\.0\.out_layers\.3",
            "middle_block.res1.conv_out",
        ),
        (r"middle_block\.1", "middle_block.transformer"),
        (
            r"middle_block\.2\.in_layers\.0",
            "middle_block.res2.norm_in",
        ),
        (
            r"middle_block\.2\.in_layers\.2",
            "middle_block.res2.conv_in",
        ),
        (
            r"middle_block\.2\.emb_layers\.1",
            "middle_block.res2.lin_embed",
        ),
        (
            r"middle_block\.2\.out_layers\.0",
            "middle_block.res2.norm_out",
        ),
        (
            r"middle_block\.2\.out_layers\.3",
            "middle_block.res2.conv_out",
        ),
        // res transformer
        (r"middle_block\.1", "$1.transformer"),
        (r"(rt\d+|rtu\d+)\.1", "$1.transformer"),
        (r"(.*?).transformer_blocks.0", "$1.transformer"),
        // cross attention
        (r"transformer.attn(\d+)\.to_q", "transformer.attn$1.query"),
        (r"transformer.attn(\d+)\.to_k", "transformer.attn$1.key"),
        (r"transformer.attn(\d+)\.to_v", "transformer.attn$1.value"),
        (r"transformer.attn(\d+)\.to_out.0", "transformer.attn$1.out"),
        // feed forward
        (r"transformer.ff\.net.0.proj", "transformer.mlp.geglu.proj"),
        (r"transformer.ff\.net.2", "transformer.mlp.lin"),
        // convert norm weight -> norm gamma
        ("norm(.*?).weight", "norm$1.gamma"),
        ("norm(.*?).bias", "norm$1.beta"),
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
        (r"first_stage_model\.decoder\.norm_out", "decoder.norm_out"),
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
            r"first_stage_model\.decoder\.mid\.attn_1\.norm",
            "decoder.mid.attn.norm",
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
        (r"first_stage_model\.encoder\.norm_out", "encoder.norm_out"),
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
            r"first_stage_model\.encoder\.mid\.attn_1\.norm",
            "encoder.mid.attn.norm",
        ),
        (
            r"first_stage_model\.encoder\.mid\.attn_1\.(q|k|v|proj_out)",
            "encoder.mid.attn.$1",
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
        eprintln!("Usage: {} <checkpoint> <model_name>", args[0]);
        process::exit(1);
    }

    let checkpoint = &args[1];
    let model_name = &args[2];

    match load_pretrained::<Backend>(checkpoint, &device) {
        Ok(model) => {
            println!("Saving burnpack...");

            let mut store = BurnpackStore::from_file(&model_name)
                .overwrite(true)
                .metadata("format", "safetensor")
                .metadata("description", "Sample file for examining Burnpack format")
                .metadata("version", env!("CARGO_PKG_VERSION"))
                .metadata("author", "Burn Example");
            model.save_into(&mut store).expect("Failed to save model");
        }
        Err(e) => {
            eprintln!("Failed to convert dump to model: {:?}", e);
            process::exit(1);
        }
    }
    println!("Successfully converted {} to {}", checkpoint, model_name);
}
