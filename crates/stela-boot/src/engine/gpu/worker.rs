//! The compile workers the GPUs' jobs and Linux work go to: they run in
//! [`crate::engine::workers`], which also translates guest code ahead of
//! execution on them.
pub(in crate::engine) use crate::engine::workers::{present, wake};
