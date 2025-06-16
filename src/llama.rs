use crate::{layer, Error, Result, Tensor, F};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    hidden_size: i64,
    num_heads: i64,
    num_kv_heads: i64,
    ffn_size: i64,
    norm_eps: f32,
    num_layers: i64,
    vocab_size: i64,
    max_ctx_length: i64,
    has_qkv_proj_bias: bool,
}

pub struct MLP {
    gate_up_proj: layer::Linear,
    down_proj: layer::Linear,
}

impl MLP {
    pub fn from_builder(builder: &layer::Builder) -> Result<Self> {
        let gate_up_proj = layer::Linear::from_builder(true, &builder.pp("gate_up_proj"))?;
        let down_proj = layer::Linear::from_builder(true, &builder.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.gate_up_proj.forward(x)?;
        let x = F::swiglu(&x)?;
        let x = self.down_proj.forward(&x)?;

        Ok(x)
    }
}
