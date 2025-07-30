/// Model inference performance details.
use {
    crate::query::Query,
    rknpu2_sys::{_rknn_query_cmd, rknn_perf_detail},
};

pub struct PerfDetail {
    inner: rknn_perf_detail,
}

impl Query for PerfDetail {
    const QUERY_TYPE: _rknn_query_cmd::Type = _rknn_query_cmd::RKNN_QUERY_PERF_DETAIL;

    type Output = rknn_perf_detail;
}

impl From<rknn_perf_detail> for PerfDetail {
    fn from(inner: rknn_perf_detail) -> Self {
        Self { inner }
    }
}

impl PerfDetail {
    pub fn details(&self) -> &str {
        if self.inner.data_len == 0 {
            return "";
        }

        if self.inner.perf_data.is_null() {
            return "";
        }

        let s = unsafe {
            std::slice::from_raw_parts(self.inner.perf_data, self.inner.data_len as usize)
        };
        unsafe { std::str::from_utf8_unchecked(s) }
    }
}
