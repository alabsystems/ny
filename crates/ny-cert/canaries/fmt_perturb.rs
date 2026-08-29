//! PERTURBATION: identical shape, but no Formatter method is called.
//! If THIS proves, write_str is genuinely the blocker. If it also fails,
//! the fmt cluster is blocked by something else and write_str is a red herring.
#![allow(dead_code)]
use std::fmt;
pub enum BranchError { Empty, Bad }
impl fmt::Debug for BranchError {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result { Ok(()) }
}
