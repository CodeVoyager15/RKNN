/// Model inference duration.
use rknpu2_sys::{_rknn_query_cmd, rknn_perf_run};

use crate::query::Query;

pub struct PerfRun {
    inner: rknn_perf_run,
}

impl Query for PerfRun {
    const QUERY_TYPE: _rknn_query_cmd::Type = _rknn_query_cmd::RKNN_QUERY_PERF_RUN;

    type Output = rknn_perf_run;
}

impl From<rknn_perf_run> for PerfRun {
    fn from(value: rknn_perf_run) -> Self {
        PerfRun { inner: value }
    }
}

impl PerfRun {
    pub fn run_duration(&self) -> i64 {
        self.inner.run_duration
    }
}
