use std::fs::File;

use crate::Result;
use llama2::{config::Config, state::State, weights::TransformerWeights};

pub struct Llama2Model {
    pub config: Config,
    weights: TransformerWeights,
    state: State,
}

impl Llama2Model {
    pub fn new(checkpoint_path: &str) -> Result<Self> {
        let mut file = File::open(checkpoint_path)?;

        let config = Config::from_reader(&mut file)?;
        println!("{config:#?}");

        let weights = TransformerWeights::from_reader(&mut file, &config)?;
        // println!("{weights:#?}");

        let state = State::new(&config);

        Ok(Self {
            config,
            weights,
            state,
        })
    }
}
