#[cfg(any(feature = "rk3576", feature = "rk35xx"))]
use crate::{
    query::InputOutputNum,
    tensor::{TensorType, builder::TensorBuilder, tensor::Tensor},
};
use {
    crate::{
        Error,
        query::{Query, QueryWithInput},
    },
    rknpu2_sys::{rknn_context, rknn_init},
    std::{ffi::c_void, ptr},
};

pub struct RKNN {
    pub(crate) ctx: rknn_context,
}

impl RKNN {
    pub fn new(model_data: &mut [u8], flags: u32) -> Result<Self, Error> {
        let mut ctx: rknn_context = 0;
        let ret = unsafe {
            rknn_init(
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
        Ok(Self { ctx })
    }

    pub fn query<T: Query>(&self) -> Result<T, Error> {
        let mut result = std::mem::MaybeUninit::<T::Output>::uninit();
        let ret = unsafe {
            rknpu2_sys::rknn_query(
                self.ctx,
                T::QUERY_TYPE,
                &mut result as *mut _ as *mut c_void,
                std::mem::size_of::<T::Output>() as u32,
            )
        };
        if ret != 0 {
            return Err(ret.into());
        }
        unsafe { Ok(result.assume_init().into()) }
    }

    pub fn query_with_input<T: QueryWithInput>(&self, input: T::Input) -> Result<T, Error> {
        let mut result = std::mem::MaybeUninit::<T::Output>::uninit();

        // SAFETY: we are immediately initializing the memory via `prepare`.
        T::prepare(input, unsafe { &mut *result.as_mut_ptr() });

        let ret = unsafe {
            rknpu2_sys::rknn_query(
                self.ctx,
                T::QUERY_TYPE,
                result.as_mut_ptr() as *mut _ as *mut c_void,
                std::mem::size_of::<T::Output>() as u32,
            )
        };
        if ret != 0 {
            return Err(ret.into());
        }
        unsafe { Ok(result.assume_init().into()) }
    }

    pub fn run(&self) -> Result<(), Error> {
        let ret = unsafe { rknpu2_sys::rknn_run(self.ctx, ptr::null_mut()) };
        if ret != 0 {
            return Err(ret.into());
        }
        Ok(())
    }

    #[cfg(any(feature = "rk3576", feature = "rk35xx"))]
    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    pub fn set_inputs<T: TensorType>(&self, tensors: &[Tensor<T>]) -> Result<(), Error> {
        let mut ffi_inputs: Vec<rknpu2_sys::rknn_input> =
            tensors.iter().map(|t| t.as_input()).collect();

        let ret = unsafe {
            rknpu2_sys::rknn_inputs_set(self.ctx, ffi_inputs.len() as u32, ffi_inputs.as_mut_ptr())
        };

        if ret != 0 {
            return Err(ret.into());
        }

        Ok(())
    }

    #[cfg(any(feature = "rk3576", feature = "rk35xx"))]
    #[cfg_attr(
        feature = "docs",
        doc(cfg(any(feature = "rk35xx", feature = "rk3576")))
    )]
    pub fn get_outputs<T: TensorType + Copy>(&self) -> Result<Vec<Tensor<T>>, Error> {
        let mut outputs = Vec::new();
        let num = self.query::<InputOutputNum>()?;
        for i in 0..num.output_num() {
            let output = TensorBuilder::new_output(&self, i).allocate()?;

            outputs.push(output);
        }

        let mut outputs_ffi = outputs
            .iter_mut()
            .map(|t| t.as_output())
            .collect::<Vec<_>>();

        let ret = unsafe {
            rknpu2_sys::rknn_outputs_get(
                self.ctx,
                outputs_ffi.len() as u32,
                outputs_ffi.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        if ret != 0 {
            return Err(ret.into());
        }

        Ok(outputs)
    }
}

impl Drop for RKNN {
    fn drop(&mut self) {
        unsafe {
            rknpu2_sys::rknn_destroy(self.ctx);
        }
    }
}
