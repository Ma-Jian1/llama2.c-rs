use thiserror::Error;

#[derive(Error, Debug)]
pub enum LlamaError {
    #[error("failed to read")]
    Reader(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, LlamaError>;
