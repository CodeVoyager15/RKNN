use {
    crate::{Error, RKNN, api::RKNNAPI},
    rknpu2_sys::rknn_context,
    std::{ffi::c_void, ptr},
};

/// Trait used for linked
pub struct LinkedAPI;

impl RKNNAPI for LinkedAPI {
    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn dup_context(
        &self,
        context_in: *mut rknpu2_sys::rknn_context,
        context_out: *mut rknpu2_sys::rknn_context,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_dup_context(context_in, context_out) };
        Ok(ret)
    }

    unsafe fn destroy(
        &self,
        context: rknpu2_sys::rknn_context,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_destroy(context) };
        Ok(ret)
    }

    unsafe fn query(
        &self,
        context: rknpu2_sys::rknn_context,
        cmd: rknpu2_sys::rknn_query_cmd,
        info: *mut std::ffi::c_void,
        size: u32,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_query(context, cmd, info, size) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn inputs_set(
        &self,
        context: rknpu2_sys::rknn_context,
        n_inputs: u32,
        inputs: *mut rknpu2_sys::rknn_input,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_inputs_set(context, n_inputs, inputs) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn set_batch_core_num(
        &self,
        context: rknpu2_sys::rknn_context,
        core_num: std::ffi::c_int,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_set_batch_core_num(context, core_num) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn set_core_mask(
        &self,
        context: rknpu2_sys::rknn_context,
        core_mask: rknpu2_sys::rknn_core_mask,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_set_core_mask(context, core_mask) };
        Ok(ret)
    }

    unsafe fn run(
        &self,
        context: rknpu2_sys::rknn_context,
        extend: *mut rknpu2_sys::rknn_run_extend,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_run(context, extend) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn outputs_get(
        &self,
        context: rknpu2_sys::rknn_context,
        n_outputs: u32,
        outputs: *mut rknpu2_sys::rknn_output,
        extend: *mut rknpu2_sys::rknn_output_extend,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_outputs_get(context, n_outputs, outputs, extend) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn outputs_release(
        &self,
        context: rknpu2_sys::rknn_context,
        n_outputs: u32,
        outputs: *mut rknpu2_sys::rknn_output,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_outputs_release(context, n_outputs, outputs) };
        Ok(ret)
    }

    unsafe fn create_mem_from_phys(
        &self,
        ctx: rknpu2_sys::rknn_context,
        phys_addr: u64,
        virt_addr: *mut std::ffi::c_void,
        size: u32,
    ) -> Result<*mut rknpu2_sys::rknn_tensor_mem, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_create_mem_from_phys(ctx, phys_addr, virt_addr, size) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576", feature = "rv110x")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576", feature = "rv110x"))]
    unsafe fn create_mem_from_fd(
        &self,
        ctx: rknpu2_sys::rknn_context,
        fd: i32,
        virt_addr: *mut std::ffi::c_void,
        size: u32,
        offset: i32,
    ) -> Result<*mut rknpu2_sys::rknn_tensor_mem, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_create_mem_from_fd(ctx, fd, virt_addr, size, offset) };
        Ok(ret)
    }

    unsafe fn create_mem(
        &self,
        ctx: rknpu2_sys::rknn_context,
        size: u32,
    ) -> Result<*mut rknpu2_sys::rknn_tensor_mem, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_create_mem(ctx, size) };
        Ok(ret)
    }

    unsafe fn create_mem2(
        &self,
        ctx: rknpu2_sys::rknn_context,
        size: u64,
        alloc_flags: u64,
    ) -> Result<*mut rknpu2_sys::rknn_tensor_mem, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_create_mem2(ctx, size, alloc_flags) };
        Ok(ret)
    }

    unsafe fn destroy_mem(
        &self,
        ctx: rknpu2_sys::rknn_context,
        mem: *mut rknpu2_sys::rknn_tensor_mem,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_destroy_mem(ctx, mem) };
        Ok(ret)
    }

    unsafe fn set_weight_mem(
        &self,
        ctx: rknpu2_sys::rknn_context,
        mem: *mut rknpu2_sys::rknn_tensor_mem,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_set_weight_mem(ctx, mem) };
        Ok(ret)
    }

    unsafe fn set_internal_mem(
        &self,
        ctx: rknpu2_sys::rknn_context,
        mem: *mut rknpu2_sys::rknn_tensor_mem,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_set_internal_mem(ctx, mem) };
        Ok(ret)
    }

    unsafe fn set_io_mem(
        &self,
        ctx: rknpu2_sys::rknn_context,
        mem: *mut rknpu2_sys::rknn_tensor_mem,
        attr: *mut rknpu2_sys::rknn_tensor_attr,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_set_io_mem(ctx, mem, attr) };
        Ok(ret)
    }

    unsafe fn set_input_shape(
        &self,
        ctx: rknpu2_sys::rknn_context,
        attr: *mut rknpu2_sys::rknn_tensor_attr,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_set_input_shape(ctx, attr) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn set_input_shapes(
        &self,
        ctx: rknpu2_sys::rknn_context,
        n_inputs: u32,
        attr: *mut rknpu2_sys::rknn_tensor_attr,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_set_input_shapes(ctx, n_inputs, attr) };
        Ok(ret)
    }

    unsafe fn mem_sync(
        &self,
        context: rknpu2_sys::rknn_context,
        mem: *mut rknpu2_sys::rknn_tensor_mem,
        mode: rknpu2_sys::rknn_mem_sync_mode,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_mem_sync(context, mem, mode) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_create(
        &self,
        ctx: *mut rknpu2_sys::rknn_matmul_ctx,
        info: *mut rknpu2_sys::rknn_matmul_info,
        io_attr: *mut rknpu2_sys::rknn_matmul_io_attr,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_matmul_create(ctx, info, io_attr) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_create_dynamic_shape(
        &self,
        ctx: *mut rknpu2_sys::rknn_matmul_ctx,
        info: *mut rknpu2_sys::rknn_matmul_info,
        shape_num: std::ffi::c_int,
        dynamic_shapes: *mut rknpu2_sys::rknn_matmul_shape,
        io_attrs: *mut rknpu2_sys::rknn_matmul_io_attr,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe {
            rknpu2_sys::rknn_matmul_create_dynamic_shape(
                ctx,
                info,
                shape_num,
                dynamic_shapes,
                io_attrs,
            )
        };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_set_io_mem(
        &self,
        ctx: rknpu2_sys::rknn_matmul_ctx,
        mem: *mut rknpu2_sys::rknn_tensor_mem,
        attr: *mut rknpu2_sys::rknn_matmul_tensor_attr,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_matmul_set_io_mem(ctx, mem, attr) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_set_core_mask(
        &self,
        ctx: rknpu2_sys::rknn_matmul_ctx,
        core_mask: rknpu2_sys::rknn_core_mask,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_matmul_set_core_mask(ctx, core_mask) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_set_quant_params(
        &self,
        ctx: rknpu2_sys::rknn_matmul_ctx,
        params: *mut rknpu2_sys::rknn_quant_params,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_matmul_set_quant_params(ctx, params) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_get_quant_params(
        &self,
        ctx: rknpu2_sys::rknn_matmul_ctx,
        params: *mut rknpu2_sys::rknn_quant_params,
        scale: *mut f32,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_matmul_get_quant_params(ctx, params, scale) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_set_dynamic_shape(
        &self,
        ctx: rknpu2_sys::rknn_matmul_ctx,
        shape: *mut rknpu2_sys::rknn_matmul_shape,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_matmul_set_dynamic_shape(ctx, shape) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_run(
        &self,
        ctx: rknpu2_sys::rknn_matmul_ctx,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_matmul_run(ctx) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn matmul_destroy(
        &self,
        ctx: rknpu2_sys::rknn_matmul_ctx,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_matmul_destroy(ctx) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn B_normal_layout_to_native_layout(
        &self,
        B_input: *mut std::ffi::c_void,
        B_output: *mut std::ffi::c_void,
        K: std::ffi::c_int,
        N: std::ffi::c_int,
        info: *mut rknpu2_sys::rknn_matmul_info,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe {
            rknpu2_sys::rknn_B_normal_layout_to_native_layout(B_input, B_output, K, N, info)
        };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn register_custom_ops(
        &self,
        ctx: rknpu2_sys::rknn_context,
        ops: *mut rknpu2_sys::rknn_custom_op,
        custom_op_num: u32,
    ) -> Result<std::ffi::c_int, crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_register_custom_ops(ctx, ops, custom_op_num) };
        Ok(ret)
    }

    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    #[cfg(any(feature = "rk35xx", feature = "rk3576"))]
    unsafe fn custom_op_get_op_attr(
        &self,
        op_ctx: *mut rknpu2_sys::rknn_custom_op_context,
        attr_name: *const std::ffi::c_char,
        op_attr: *mut rknpu2_sys::rknn_custom_op_attr,
    ) -> Result<(), crate::Error> {
        let ret = unsafe { rknpu2_sys::rknn_custom_op_get_op_attr(op_ctx, attr_name, op_attr) };
        Ok(ret)
    }
}

impl RKNN<LinkedAPI> {
    pub fn new(model_data: &mut [u8], flags: u32) -> Result<Self, Error> {
        let mut ctx: rknn_context = 0;
        let ret = unsafe {
            rknpu2_sys::rknn_init(
                &mut ctx as *mut _,
                model_data.as_mut_ptr() as *mut c_void,
                model_data.len() as u32,
                flags,
                ptr::null_mut(),
            )
        };
        if ret != 0 {
            return Err(ret.into());
        }
        Ok(Self {
            ctx,
            api: LinkedAPI,
        })
    }
}
