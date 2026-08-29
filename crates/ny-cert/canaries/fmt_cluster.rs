//! Minimal reproduction of ny-cert's 26-function fmt cluster: a derived-shape
//! Debug impl whose only obligation is the call to Formatter::write_str.
#![allow(dead_code)]
use std::fmt;

pub enum BranchError { Empty, Bad }

impl fmt::Debug for BranchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BranchError::Empty => f.write_str("Empty"),
            BranchError::Bad => f.write_str("Bad"),
        }
    }
}
