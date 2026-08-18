// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use prost::Message;

#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(string, tag = "3")]
    pub producer_version: String,
    #[prost(string, tag = "4")]
    pub domain: String,
    #[prost(int64, tag = "5")]
    pub model_version: i64,
    #[prost(string, tag = "6")]
    pub doc_string: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
}

#[derive(Clone, PartialEq, Eq, Message)]
pub struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}

#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    /// Sparse initializers are modeled so the loader can reject them before
    /// silently changing graph semantics. NY does not yet lower sparse tensor
    /// storage.
    #[prost(message, repeated, tag = "15")]
    pub sparse_initializer: Vec<SparseTensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
    #[cfg(feature = "onnx-value-info")]
    #[prost(message, repeated, tag = "13")]
    pub value_info: Vec<ValueInfoProto>,
}

impl GraphProto {
    /// Returns value_info entries when onnx-value-info is enabled; otherwise an empty slice.
    pub fn value_info(&self) -> &[ValueInfoProto] {
        #[cfg(feature = "onnx-value-info")]
        {
            &self.value_info
        }
        #[cfg(not(feature = "onnx-value-info"))]
        {
            &[]
        }
    }
}

/// Deprecated: use GraphProto::value_info instead.
#[deprecated(note = "Use GraphProto::value_info")]
pub fn graph_value_info(graph: &GraphProto) -> &[ValueInfoProto] {
    graph.value_info()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "onnx-value-info")]
    #[test]
    fn graph_value_info_returns_entries() {
        let info = ValueInfoProto {
            name: "value".to_string(),
            r#type: None,
        };
        let graph = GraphProto {
            node: Vec::new(),
            name: String::new(),
            initializer: Vec::new(),
            sparse_initializer: Vec::new(),
            input: Vec::new(),
            output: Vec::new(),
            value_info: vec![info.clone()],
        };
        let info_slice = graph.value_info();
        assert_eq!(info_slice.len(), 1);
        assert_eq!(info_slice[0].name, info.name);
    }

    #[cfg(not(feature = "onnx-value-info"))]
    #[test]
    fn graph_value_info_is_empty_without_feature() {
        let graph = GraphProto {
            node: Vec::new(),
            name: String::new(),
            initializer: Vec::new(),
            sparse_initializer: Vec::new(),
            input: Vec::new(),
            output: Vec::new(),
        };
        assert!(graph.value_info().is_empty());
    }
}

/// ONNX sparse tensor storage.
#[derive(Clone, PartialEq, Message)]
pub struct SparseTensorProto {
    #[prost(message, optional, tag = "1")]
    pub values: Option<TensorProto>,
    #[prost(message, optional, tag = "2")]
    pub indices: Option<TensorProto>,
    #[prost(int64, repeated, tag = "3")]
    pub dims: Vec<i64>,
}

#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(string, tag = "7")]
    pub domain: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    /// Deprecated partial-tensor segment metadata. NY models only complete
    /// tensors and rejects this field when present.
    #[prost(message, optional, tag = "3")]
    pub segment: Option<TensorSegmentProto>,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,
    // Typed non-raw payload fields. The ONNX spec stores tensor data EITHER in
    // raw_data OR in exactly one of these repeated fields depending on
    // data_type (float_data: FLOAT/COMPLEX; int32_data: INT8/UINT8/INT16/
    // UINT16/INT32/BOOL/FLOAT16/BFLOAT16/FLOAT8/INT4/UINT4; int64_data: INT64;
    // double_data: DOUBLE). prost silently drops unknown tags, so every field
    // a decoder consults must be declared here or its payload vanishes.
    #[prost(float, repeated, tag = "4")]
    pub float_data: Vec<f32>,
    #[prost(int32, repeated, tag = "5")]
    pub int32_data: Vec<i32>,
    #[prost(int64, repeated, tag = "7")]
    pub int64_data: Vec<i64>,
    #[prost(double, repeated, tag = "10")]
    pub double_data: Vec<f64>,
    #[prost(bytes = "vec", repeated, tag = "6")]
    pub string_data: Vec<Vec<u8>>,
    #[prost(uint64, repeated, tag = "11")]
    pub uint64_data: Vec<u64>,
    /// Key/value metadata for an external raw-data payload.
    #[prost(message, repeated, tag = "13")]
    pub external_data: Vec<StringStringEntryProto>,
    /// TensorProto.DataLocation: 0 = DEFAULT (inline), 1 = EXTERNAL.
    #[prost(int32, tag = "14")]
    pub data_location: i32,
}

/// Deprecated TensorProto segment range.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TensorSegmentProto {
    #[prost(int64, tag = "1")]
    pub begin: i64,
    #[prost(int64, tag = "2")]
    pub end: i64,
}

/// ONNX's protobuf-compatible string map entry.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct StringStringEntryProto {
    #[prost(string, tag = "1")]
    pub key: String,
    #[prost(string, tag = "2")]
    pub value: String,
}

#[derive(Clone, PartialEq, Eq, Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
}

#[derive(Clone, PartialEq, Eq, Message)]
pub struct TypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<TensorTypeProto>,
}

#[derive(Clone, PartialEq, Eq, Message)]
pub struct TensorTypeProto {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, Eq, Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<tensor_shape_proto::Dimension>,
}

pub mod tensor_shape_proto {
    use prost::Message;

    #[derive(Clone, PartialEq, Eq, Message)]
    pub struct Dimension {
        #[prost(oneof = "dimension::Value", tags = "1, 2")]
        pub value: Option<dimension::Value>,
    }

    pub mod dimension {
        #[derive(Clone, PartialEq, Eq, prost::Oneof)]
        pub enum Value {
            #[prost(int64, tag = "1")]
            DimValue(i64),
            #[prost(string, tag = "2")]
            DimParam(String),
        }
    }
}

#[derive(Clone, PartialEq, Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    /// Attribute reference used only by nodes in a FunctionProto body. A
    /// reference carries no local payload; graph attributes in a ModelProto
    /// must not use it.
    #[prost(string, tag = "21")]
    pub ref_attr_name: String,
    #[prost(string, tag = "13")]
    pub doc_string: String,
    #[prost(int32, tag = "20")]
    pub r#type: i32,
    // These scalar fields are optional so an explicitly encoded inactive
    // default remains distinguishable from an absent wire tag. ONNX permits a
    // selected scalar at its proto3 default to omit the payload tag.
    #[prost(float, optional, tag = "2")]
    pub f: Option<f32>,
    #[prost(int64, optional, tag = "3")]
    pub i: Option<i64>,
    #[prost(bytes = "vec", optional, tag = "4")]
    pub s: Option<Vec<u8>>,
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,
    #[prost(message, optional, tag = "6")]
    pub g: Option<GraphProto>,
    #[prost(message, optional, tag = "22")]
    pub sparse_tensor: Option<SparseTensorProto>,
    #[prost(message, optional, tag = "14")]
    pub tp: Option<TypeProto>,
    #[prost(float, repeated, tag = "7")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
    #[prost(bytes = "vec", repeated, tag = "9")]
    pub strings: Vec<Vec<u8>>,
    #[prost(message, repeated, tag = "10")]
    pub tensors: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub graphs: Vec<GraphProto>,
    #[prost(message, repeated, tag = "23")]
    pub sparse_tensors: Vec<SparseTensorProto>,
    #[prost(message, repeated, tag = "15")]
    pub type_protos: Vec<TypeProto>,
}

impl AttributeProto {
    /// Returns the validated FLOAT payload.
    ///
    /// Proto3 may omit the selected scalar's wire tag when its value is zero;
    /// ONNX defines that omission as the corresponding scalar default.
    pub fn f_value(&self) -> f32 {
        self.f.unwrap_or_default()
    }

    /// Returns the validated INT payload, including ONNX's encoded-default zero
    /// when the selected scalar's proto3 wire tag is absent.
    pub fn i_value(&self) -> i64 {
        self.i.unwrap_or_default()
    }

    /// Returns the validated STRING payload. As with numeric scalars, an
    /// omitted selected field denotes its proto3 default (the empty string).
    pub fn s_value(&self) -> &[u8] {
        self.s.as_deref().unwrap_or_default()
    }
}

pub mod attribute_proto {
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
    #[repr(i32)]
    pub enum AttributeType {
        Undefined = 0,
        Float = 1,
        Int = 2,
        String = 3,
        Tensor = 4,
        Graph = 5,
        Floats = 6,
        Ints = 7,
        Strings = 8,
        Tensors = 9,
        Graphs = 10,
        SparseTensor = 11,
        SparseTensors = 12,
        TypeProto = 13,
        TypeProtos = 14,
    }

    impl AttributeType {
        pub const fn as_i32(self) -> i32 {
            self as i32
        }
    }
}

pub mod attribute_type {
    use super::attribute_proto::AttributeType;

    pub const FLOAT: i32 = AttributeType::Float as i32;
    pub const INT: i32 = AttributeType::Int as i32;
    pub const STRING: i32 = AttributeType::String as i32;
    pub const TENSOR: i32 = AttributeType::Tensor as i32;
    pub const GRAPH: i32 = AttributeType::Graph as i32;
    pub const FLOATS: i32 = AttributeType::Floats as i32;
    pub const INTS: i32 = AttributeType::Ints as i32;
    pub const STRINGS: i32 = AttributeType::Strings as i32;
    pub const TENSORS: i32 = AttributeType::Tensors as i32;
    pub const GRAPHS: i32 = AttributeType::Graphs as i32;
    pub const SPARSE_TENSOR: i32 = AttributeType::SparseTensor as i32;
    pub const SPARSE_TENSORS: i32 = AttributeType::SparseTensors as i32;
    pub const TYPE_PROTO: i32 = AttributeType::TypeProto as i32;
    pub const TYPE_PROTOS: i32 = AttributeType::TypeProtos as i32;
}
