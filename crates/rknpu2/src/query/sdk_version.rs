use std::ffi::CStr;

/// Sdk and driver version information.
use rknpu2_sys::{
    _rknn_query_cmd::{RKNN_QUERY_SDK_VERSION, Type},
    rknn_sdk_version,
};

use crate::query::Query;

pub struct SdkVersion {
    pub(crate) inner: rknn_sdk_version,
}

impl SdkVersion {
    pub fn api_version(&self) -> String {
        let cstr = unsafe { CStr::from_ptr(self.inner.api_version.as_ptr()) };
        cstr.to_string_lossy().into_owned()
    }

    pub fn driver_version(&self) -> String {
        let cstr = unsafe { CStr::from_ptr(self.inner.api_version.as_ptr()) };
        cstr.to_string_lossy().into_owned()
    }
}

impl Query for SdkVersion {
    const QUERY_TYPE: Type = RKNN_QUERY_SDK_VERSION;

    type Output = rknn_sdk_version;
}

impl From<rknn_sdk_version> for SdkVersion {
    fn from(value: rknn_sdk_version) -> Self {
        SdkVersion { inner: value }
    }
}
