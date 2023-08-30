use thiserror::Error;

#[derive(Error, Debug)]
pub enum Llama2Error {
    #[error("failed to read: {0}")]
    Reader(#[from] std::io::Error),

    #[error("{0}")]
    Utf8Error(#[from] std::str::Utf8Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Llama2Error>;
