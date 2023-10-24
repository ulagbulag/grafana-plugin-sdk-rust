//! Serialization of [`Frame`]s to the JSON format.
use std::{cell::RefCell, collections::BTreeMap};

use arrow::{
    array::{types, Array, BooleanArray, PrimitiveArray, StringArray},
    datatypes::{ArrowPrimitiveType, DataType, TimeUnit},
    temporal_conversions::MILLISECONDS_IN_DAY,
};
use num_traits::Float;
use serde::{
    ser::{Error, SerializeMap, SerializeSeq},
    Deserialize, Serialize, Serializer,
};
use serde_with::skip_serializing_none;

use crate::data::{
    field::{Field, FieldConfig, SimpleType, TypeInfo},
    frame::{Frame, Metadata},
};

impl Serialize for Frame {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let schema_fields: Vec<_> = self
            .fields
            .iter()
            .map(|f| {
                Ok(SerializableField {
                    name: f.name.as_str(),
                    labels: &f.labels,
                    config: f.config.as_ref(),
                    type_: f.type_info.simple_type(),
                    type_info: &f.type_info,
                })
            })
            .collect::<Result<_, _>>()?;
        let ser = SerializableFrame {
            schema: Some(SerializableFrameSchema {
                name: &self.name,
                ref_id: self.ref_id.as_deref(),
                meta: &self.meta,
                fields: &schema_fields,
            }),
            data: Some(SerializableFrameData {
                fields: &self.fields,
            }),
        };
        ser.serialize(serializer)
    }
}

#[skip_serializing_none]
#[derive(Debug, Serialize)]
pub(super) struct SerializableFrame<'a> {
    pub(super) schema: Option<SerializableFrameSchema<'a>>,
    pub(super) data: Option<SerializableFrameData<'a>>,
}

#[skip_serializing_none]
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct SerializableFrameSchema<'a> {
    pub(super) name: &'a str,
    pub(super) ref_id: Option<&'a str>,
    pub(super) meta: &'a Option<Metadata>,
    pub(super) fields: &'a [SerializableField<'a>],
}

#[skip_serializing_none]
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct SerializableField<'a> {
    pub name: &'a str,
    pub labels: &'a BTreeMap<String, String>,
    pub config: Option<&'a FieldConfig>,
    #[serde(rename = "type")]
    pub type_: SimpleType,
    pub type_info: &'a TypeInfo,
}

#[derive(Debug)]
pub(super) struct SerializableFrameData<'a> {
    pub(super) fields: &'a [Field],
}

impl<'a> Serialize for SerializableFrameData<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let n_fields = self.fields.len();

        let entities: Vec<_> = std::iter::repeat(RefCell::new(None))
            .take(n_fields)
            .collect();

        let values = SerializableFrameDataValues {
            fields: self.fields,
            entities: &entities,
        };

        let mut map = serializer.serialize_map(None)?;
        map.serialize_key("values")?;
        map.serialize_value(&values)?;

        if entities.iter().any(|r| r.borrow().is_some()) {
            map.serialize_key("entities")?;
            map.serialize_value(&entities)?;
        }

        map.end()
    }
}

struct SerializableFrameDataValues<'a, 'b> {
    fields: &'a [Field],
    entities: &'b [RefCell<Option<Entities>>],
}

impl<'a, 'b> Serialize for SerializableFrameDataValues<'a, 'b> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.fields.len()))?;
        for (arr, e) in self.fields.iter().zip(self.entities.iter()) {
            seq.serialize_element(&SerializableArray(&*arr.values, e))?;
        }
        seq.end()
    }
}

struct SerializableArray<'a>(&'a dyn Array, &'a RefCell<Option<Entities>>);

impl<'a> Serialize for SerializableArray<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let array = self.0;
        let len = array.len();
        match array.data_type() {
            DataType::Null => {
                serializer.collect_seq(std::iter::repeat::<Option<()>>(None).take(len))
            }
            DataType::Boolean => serializer.collect_seq(
                array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .unwrap()
                    .iter(),
            ),
            DataType::Utf8 | DataType::LargeUtf8 => {
                serializer.collect_seq(array.as_any().downcast_ref::<StringArray>().unwrap().iter())
            }
            DataType::Int8 => {
                serializer.collect_seq(primitive_array_iter::<types::Int8Type>(array))
            }
            DataType::Int16 => {
                serializer.collect_seq(primitive_array_iter::<types::Int16Type>(array))
            }
            DataType::Int32 => {
                serializer.collect_seq(primitive_array_iter::<types::Int32Type>(array))
            }
            DataType::Int64 => {
                serializer.collect_seq(primitive_array_iter::<types::Int64Type>(array))
            }
            DataType::Date32 => serializer.collect_seq(
                primitive_array_iter::<types::Date32Type>(array)
                    .map(|opt| opt.map(|x| i64::from(x) * MILLISECONDS_IN_DAY)),
            ),
            DataType::Date64 => {
                serializer.collect_seq(primitive_array_iter::<types::Date64Type>(array))
            }
            DataType::Timestamp(TimeUnit::Second, _) => {
                // Timestamps should be serialized to JSON as milliseconds.
                serializer.collect_seq(
                    primitive_array_iter::<types::TimestampSecondType>(array)
                        .map(|opt| opt.map(|x| x * 1_000)),
                )
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                // Timestamps should be serialized to JSON as milliseconds.
                serializer.collect_seq(primitive_array_iter::<types::TimestampMillisecondType>(
                    array,
                ))
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                // Timestamps should be serialized to JSON as milliseconds.
                serializer.collect_seq(
                    primitive_array_iter::<types::TimestampMicrosecondType>(array)
                        .map(|opt| opt.map(|x| x / 1_000)),
                )
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                // Timestamps should be serialized to JSON as milliseconds.
                serializer.collect_seq(
                    primitive_array_iter::<types::TimestampNanosecondType>(array)
                        .map(|opt| opt.map(|x| x / 1_000_000)),
                )
            }
            DataType::UInt8 => {
                serializer.collect_seq(primitive_array_iter::<types::UInt8Type>(array))
            }
            DataType::UInt16 => {
                serializer.collect_seq(primitive_array_iter::<types::UInt16Type>(array))
            }
            DataType::UInt32 => {
                serializer.collect_seq(primitive_array_iter::<types::UInt32Type>(array))
            }
            DataType::UInt64 => {
                serializer.collect_seq(primitive_array_iter::<types::UInt64Type>(array))
            }
            DataType::Float32 => serialize_floats_and_collect_entities::<S, types::Float32Type>(
                serializer, array, self.1,
            ),
            DataType::Float64 => serialize_floats_and_collect_entities::<S, types::Float64Type>(
                serializer, array, self.1,
            ),
            _ => Err(S::Error::custom("unsupported arrow datatype")),
        }
    }
}

fn primitive_array_iter<T>(
    array: &dyn Array,
) -> impl Iterator<Item = Option<<T as ArrowPrimitiveType>::Native>> + '_
where
    T: ArrowPrimitiveType,
{
    array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .unwrap()
        .iter()
}

fn serialize_floats_and_collect_entities<S, T>(
    serializer: S,
    array: &dyn Array,
    entities_ref: &RefCell<Option<Entities>>,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: ArrowPrimitiveType,
    <T as ArrowPrimitiveType>::Native: Float + Serialize,
{
    let array = array.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    let mut seq = serializer.serialize_seq(Some(array.len()))?;
    let mut entities = Entities::default();
    for (i, el) in array.iter().enumerate() {
        seq.serialize_element(&el)?;
        match el {
            Some(x) if x.is_nan() => entities.nan.push(i),
            Some(x) if x.is_infinite() && x.is_sign_positive() => entities.inf.push(i),
            Some(x) if x.is_infinite() && x.is_sign_negative() => entities.neg_inf.push(i),
            _ => {}
        }
    }
    if !entities.nan.is_empty() || !entities.inf.is_empty() || !entities.neg_inf.is_empty() {
        *entities_ref.borrow_mut() = Some(entities);
    }
    seq.end()
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct Entities {
    #[serde(default, rename = "NaN", skip_serializing_if = "Vec::is_empty")]
    pub(crate) nan: Vec<usize>,
    #[serde(default, rename = "Inf", skip_serializing_if = "Vec::is_empty")]
    pub(crate) inf: Vec<usize>,
    #[serde(default, rename = "NegInf", skip_serializing_if = "Vec::is_empty")]
    pub(crate) neg_inf: Vec<usize>,
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow::{
        array::{types, PrimitiveArray},
        datatypes::{DataType, TimeUnit},
    };
    use pretty_assertions::assert_eq;
    use serde_json::{from_str, json, to_string, to_string_pretty};

    use crate::data::{field::*, frame::*};

    #[test]
    #[ignore]
    // Ignore this test for now, the JSON isn't minified.
    fn serialize_golden() {
        let expected = include_str!("golden.json");
        let f: Frame = from_str(expected).unwrap();
        let actual = to_string(&f).unwrap();
        assert_eq!(&actual, expected);
    }

    #[test]
    fn round_trip_small() {
        let f = Frame {
            name: "many_types".to_string(),
            ref_id: Some("A".to_string()),
            meta: Some(Metadata {
                custom: json!({"Hi": "there"}).as_object_mut().map(std::mem::take),
                ..Default::default()
            }),
            fields: vec![
                Field {
                    name: "int8_values".to_string(),
                    labels: Default::default(),
                    config: None,
                    values: Arc::new(PrimitiveArray::<types::Int8Type>::from_iter_values([
                        -128, -128, 0, 127, 127,
                    ])),
                    type_info: TypeInfo {
                        frame: TypeInfoType::Int8,
                        nullable: Some(false),
                    },
                },
                Field {
                    name: "date32_values".to_string(),
                    labels: Default::default(),
                    config: None,
                    values: Arc::new(
                        PrimitiveArray::<types::Date32Type>::from_iter_values([
                            18895, 18896, 18897, 18898, 18899,
                        ])
                        .reinterpret_cast::<types::Date32Type>(),
                    ),
                    type_info: TypeInfo {
                        frame: TypeInfoType::Time,
                        nullable: Some(false),
                    },
                },
                Field {
                    name: "date64_values".to_string(),
                    labels: Default::default(),
                    config: None,
                    values: Arc::new(
                        PrimitiveArray::<types::Date64Type>::from_iter_values([
                            1632528000000,
                            1632614400000,
                            1632700800000,
                            1632787200000,
                            1632873600000,
                        ])
                        .reinterpret_cast::<types::Date64Type>(),
                    ),
                    type_info: TypeInfo {
                        frame: TypeInfoType::Time,
                        nullable: Some(false),
                    },
                },
                Field {
                    name: "timestamp_s_values".to_string(),
                    labels: Default::default(),
                    config: None,
                    values: Arc::new(
                        PrimitiveArray::<types::TimestampSecondType>::from_iter_values([
                            1632855151, 1632855152, 1632855153, 1632855154, 1632855155,
                        ]),
                    ),
                    type_info: TypeInfo {
                        frame: TypeInfoType::Time,
                        nullable: Some(false),
                    },
                },
                Field {
                    name: "timestamp_ms_values".to_string(),
                    labels: Default::default(),
                    config: None,
                    values: Arc::new(
                        PrimitiveArray::<types::TimestampMillisecondType>::from_iter_values([
                            1632855151000,
                            1632855152000,
                            1632855153000,
                            1632855154000,
                            1632855155000,
                        ]),
                    ),
                    type_info: TypeInfo {
                        frame: TypeInfoType::Time,
                        nullable: Some(false),
                    },
                },
                Field {
                    name: "timestamp_us_values".to_string(),
                    labels: Default::default(),
                    config: None,
                    values: Arc::new(
                        PrimitiveArray::<types::TimestampMicrosecondType>::from_iter_values([
                            1632855151000000,
                            1632855152000000,
                            1632855153000000,
                            1632855154000000,
                            1632855155000000,
                        ]),
                    ),
                    type_info: TypeInfo {
                        frame: TypeInfoType::Time,
                        nullable: Some(false),
                    },
                },
                Field {
                    name: "timestamp_ns_values".to_string(),
                    labels: Default::default(),
                    config: None,
                    values: Arc::new(
                        PrimitiveArray::<types::TimestampNanosecondType>::from_iter_values([
                            1632855151000000000,
                            1632855152000000000,
                            1632855153000000000,
                            1632855154000000000,
                            1632855155000000000,
                        ])
                        .with_data_type(DataType::Timestamp(
                            TimeUnit::Nanosecond,
                            Some("+12:00".into()),
                        )),
                    ),
                    type_info: TypeInfo {
                        frame: TypeInfoType::Time,
                        nullable: Some(false),
                    },
                },
            ],
        };
        let jdoc = to_string_pretty(&f).unwrap();
        let parsed: Frame = from_str(&jdoc).unwrap();
        let jdoc_again = to_string_pretty(&parsed).unwrap();
        // Compare the JSON reprs; the internal Arrow datatypes will
        // be different because the JSON representation is lossy
        // (we lose timestamp representations and timezones).
        assert_eq!(jdoc, jdoc_again);
    }

    #[test]
    fn round_trip_full() {
        let jdoc = include_str!("golden.json");
        let parsed: Frame = from_str(jdoc).unwrap();
        let jdoc_ser = to_string(&parsed).unwrap();
        let parsed_again: Frame = from_str(jdoc).unwrap();
        let jdoc_ser_again = to_string(&parsed_again).unwrap();
        // Compare the JSON reprs; the internal Arrow datatypes will
        // be different because the JSON representation is lossy
        // (we lose timestamp representations and timezones).
        assert_eq!(jdoc_ser, jdoc_ser_again);
    }
}
