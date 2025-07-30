/// Trait for RKNN API operations.

#[cfg(any(feature = "rk35xx", feature = "rk3576"))]
use rknpu2_sys::{
    rknn_core_mask, rknn_custom_op, rknn_custom_op_attr, rknn_custom_op_context, rknn_input,
    rknn_matmul_ctx, rknn_matmul_info, rknn_matmul_io_attr, rknn_matmul_shape,
    rknn_matmul_tensor_attr, rknn_output, rknn_output_extend, rknn_quant_params,
};
#[cfg(any(feature = "rk35xx", feature = "rk3576"))]
use std::ffi::c_char;
use {
    rknpu2_sys::{
        rknn_context, rknn_mem_sync_mode, rknn_query_cmd, rknn_run_extend, rknn_tensor_attr,
        rknn_tensor_mem,
    },
    std::ffi::{c_int, c_void},
};

#[cfg(not(feature = "libloading"))]
pub mod linked;

#[cfg(feature = "libloading")]
pub mod runtime;

use crate::Error;

pub trait RKNNAPI {
    // ───── core runtime ──────────────────────────────────────────────────────
    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn dup_context(
        &self,
        context_in: *mut rknn_context,
        context_out: *mut rknn_context,
    ) -> Result<c_int, Error>;

    unsafe fn destroy(&self, context: rknn_context) -> Result<c_int, Error>;

    unsafe fn query(
        &self,
        context: rknn_context,
        cmd: rknn_query_cmd,
        info: *mut c_void,
        size: u32,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn inputs_set(
        &self,
        context: rknn_context,
        n_inputs: u32,
        inputs: *mut rknn_input,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn set_batch_core_num(
        &self,
        context: rknn_context,
        core_num: c_int,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn set_core_mask(
        &self,
        context: rknn_context,
        core_mask: rknn_core_mask,
    ) -> Result<c_int, Error>;

    unsafe fn run(
        &self,
        context: rknn_context,
        extend: *mut rknn_run_extend,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn outputs_get(
        &self,
        context: rknn_context,
        n_outputs: u32,
        outputs: *mut rknn_output,
        extend: *mut rknn_output_extend,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn outputs_release(
        &self,
        context: rknn_context,
        n_outputs: u32,
        outputs: *mut rknn_output,
    ) -> Result<c_int, Error>;

    // ───── memory helpers ────────────────────────────────────────────────────
    unsafe fn create_mem_from_phys(
        &self,
        ctx: rknn_context,
        phys_addr: u64,
        virt_addr: *mut c_void,
        size: u32,
    ) -> Result<*mut rknn_tensor_mem, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576", feature = "rv110x")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576", feature = "rv110x"))]
    unsafe fn create_mem_from_fd(
        &self,
        ctx: rknn_context,
        fd: i32,
        virt_addr: *mut c_void,
        size: u32,
        offset: i32,
    ) -> Result<*mut rknn_tensor_mem, Error>;

    unsafe fn create_mem(
        &self,
        ctx: rknn_context,
        size: u32,
    ) -> Result<*mut rknn_tensor_mem, Error>;

    unsafe fn create_mem2(
        &self,
        ctx: rknn_context,
        size: u64,
        alloc_flags: u64,
    ) -> Result<*mut rknn_tensor_mem, Error>;

    unsafe fn destroy_mem(
        &self,
        ctx: rknn_context,
        mem: *mut rknn_tensor_mem,
    ) -> Result<c_int, Error>;

    unsafe fn set_weight_mem(
        &self,
        ctx: rknn_context,
        mem: *mut rknn_tensor_mem,
    ) -> Result<c_int, Error>;

    unsafe fn set_internal_mem(
        &self,
        ctx: rknn_context,
        mem: *mut rknn_tensor_mem,
    ) -> Result<c_int, Error>;

    unsafe fn set_io_mem(
        &self,
        ctx: rknn_context,
        mem: *mut rknn_tensor_mem,
        attr: *mut rknn_tensor_attr,
    ) -> Result<c_int, Error>;

    unsafe fn set_input_shape(
        &self,
        ctx: rknn_context,
        attr: *mut rknn_tensor_attr,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn set_input_shapes(
        &self,
        ctx: rknn_context,
        n_inputs: u32,
        attr: *mut rknn_tensor_attr,
    ) -> Result<c_int, Error>;

    unsafe fn mem_sync(
        &self,
        context: rknn_context,
        mem: *mut rknn_tensor_mem,
        mode: rknn_mem_sync_mode,
    ) -> Result<c_int, Error>;

    // ───── MatMul accelerator sub-API ────────────────────────────────────────

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_create(
        &self,
        ctx: *mut rknn_matmul_ctx,
        info: *mut rknn_matmul_info,
        io_attr: *mut rknn_matmul_io_attr,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_create_dynamic_shape(
        &self,
        ctx: *mut rknn_matmul_ctx,
        info: *mut rknn_matmul_info,
        shape_num: c_int,
        dynamic_shapes: *mut rknn_matmul_shape,
        io_attrs: *mut rknn_matmul_io_attr,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_set_io_mem(
        &self,
        ctx: rknn_matmul_ctx,
        mem: *mut rknn_tensor_mem,
        attr: *mut rknn_matmul_tensor_attr,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_set_core_mask(
        &self,
        ctx: rknn_matmul_ctx,
        core_mask: rknn_core_mask,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_set_quant_params(
        &self,
        ctx: rknn_matmul_ctx,
        params: *mut rknn_quant_params,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_get_quant_params(
        &self,
        ctx: rknn_matmul_ctx,
        params: *mut rknn_quant_params,
        scale: *mut f32,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_set_dynamic_shape(
        &self,
        ctx: rknn_matmul_ctx,
        shape: *mut rknn_matmul_shape,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_run(&self, ctx: rknn_matmul_ctx) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_destroy(&self, ctx: rknn_matmul_ctx) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn B_normal_layout_to_native_layout(
        &self,
        B_input: *mut c_void,
        B_output: *mut c_void,
        K: c_int,
        N: c_int,
        info: *mut rknn_matmul_info,
    ) -> Result<c_int, Error>;

    // ───── custom operator extension ────────────────────────────────────────

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn register_custom_ops(
        &self,
        ctx: rknn_context,
        ops: *mut rknn_custom_op,
        custom_op_num: u32,
    ) -> Result<c_int, Error>;

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn custom_op_get_op_attr(
        &self,
        op_ctx: *mut rknn_custom_op_context,
        attr_name: *const c_char,
        op_attr: *mut rknn_custom_op_attr,
    ) -> Result<(), Error>;
}
