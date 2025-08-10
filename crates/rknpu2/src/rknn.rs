use rknpu2_sys::{
    _rknn_core_mask::{
        RKNN_NPU_CORE_0, RKNN_NPU_CORE_0_1, RKNN_NPU_CORE_0_1_2, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2,
        RKNN_NPU_CORE_ALL, RKNN_NPU_CORE_AUTO,
    },
    rknn_context,
};

#[cfg(any(feature = "rk3576", feature = "rk35xx"))]
use crate::tensor::{IntoInputs, TensorT, builder::TensorBuilder};
use {
    crate::{
        Error,
        api::RKNNAPI,
        query::{Query, QueryWithInput, TensorAttrView},
    },
    std::{ffi::c_void, ptr},
};

/// Main rknn struct with ability to query the model and run inference.
pub struct RKNN<A: RKNNAPI> {
    pub(crate) ctx: rknn_context,
    pub(crate) api: A,
}

impl<A: RKNNAPI> RKNN<A> {
    pub fn query<T: Query>(&self) -> Result<T, Error> {
        let mut result = std::mem::MaybeUninit::<T::Output>::uninit();
        let ret = unsafe {
            self.api.query(
                self.ctx,
                T::QUERY_TYPE,
                &mut result as *mut _ as *mut c_void,
                std::mem::size_of::<T::Output>() as u32,
            )?
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
            self.api.query(
                self.ctx,
                T::QUERY_TYPE,
                result.as_mut_ptr() as *mut _ as *mut c_void,
                std::mem::size_of::<T::Output>() as u32,
            )?
        };
        if ret != 0 {
            return Err(ret.into());
        }
        unsafe { Ok(result.assume_init().into()) }
    }

    pub fn run(&self) -> Result<(), Error> {
        let ret = unsafe { self.api.run(self.ctx, ptr::null_mut())? };
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
    pub fn set_inputs<I: IntoInputs>(&self, tensors: I) -> Result<(), Error> {
        let tensors = tensors.into_inputs();

        let mut ffi_inputs: Vec<rknpu2_sys::rknn_input> =
            tensors.iter().map(|t| t.as_input()).collect();

        let ret = unsafe {
            self.api
                .inputs_set(self.ctx, ffi_inputs.len() as u32, ffi_inputs.as_mut_ptr())?
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
    pub fn get_outputs(&self) -> Result<Vec<TensorT>, Error> {
        use crate::{
            bf16, f16,
            query::{InputOutputNum, OutputAttr},
        };

        let mut outputs = Vec::<TensorT>::new();
        let num = self.query::<InputOutputNum>()?;
        for i in 0..num.output_num() {
            let output_attr = self.query_with_input::<OutputAttr>(i)?;

            match output_attr.dtype() {
                crate::tensor::DataTypeKind::Float32(_) => outputs.push(
                    TensorBuilder::new_output(&self, i)
                        .allocate::<f32>()?
                        .into(),
                ),
                crate::tensor::DataTypeKind::Float16(_) => outputs.push(
                    TensorBuilder::new_output(&self, i)
                        .allocate::<f16>()?
                        .into(),
                ),
                crate::tensor::DataTypeKind::BFloat16(_) => outputs.push(
                    TensorBuilder::new_output(&self, i)
                        .allocate::<bf16>()?
                        .into(),
                ),
                crate::tensor::DataTypeKind::Int4(_) => {
                    todo!("rknpu2 doesn't currently support Int4")
                }
                crate::tensor::DataTypeKind::Int8(_) => {
                    outputs.push(TensorBuilder::new_output(&self, i).allocate::<i8>()?.into())
                }
                crate::tensor::DataTypeKind::UInt8(_) => {
                    outputs.push(TensorBuilder::new_output(&self, i).allocate::<u8>()?.into())
                }
                crate::tensor::DataTypeKind::Int16(_) => outputs.push(
                    TensorBuilder::new_output(&self, i)
                        .allocate::<i16>()?
                        .into(),
                ),
                crate::tensor::DataTypeKind::UInt16(_) => outputs.push(
                    TensorBuilder::new_output(&self, i)
                        .allocate::<u16>()?
                        .into(),
                ),
                crate::tensor::DataTypeKind::Int32(_) => outputs.push(
                    TensorBuilder::new_output(&self, i)
                        .allocate::<i32>()?
                        .into(),
                ),
                crate::tensor::DataTypeKind::UInt32(_) => outputs.push(
                    TensorBuilder::new_output(&self, i)
                        .allocate::<u32>()?
                        .into(),
                ),
                crate::tensor::DataTypeKind::Int64(_) => outputs.push(
                    TensorBuilder::new_output(&self, i)
                        .allocate::<i64>()?
                        .into(),
                ),
                crate::tensor::DataTypeKind::Bool(_) => todo!(),
                crate::tensor::DataTypeKind::Max(_) => todo!(),
                crate::tensor::DataTypeKind::Other(_) => todo!(),
            }
        }

        let mut outputs_ffi = outputs
            .iter_mut()
            .map(|t| t.as_output())
            .collect::<Vec<_>>();

        let ret = unsafe {
            self.api.outputs_get(
                self.ctx,
                outputs_ffi.len() as u32,
                outputs_ffi.as_mut_ptr(),
                std::ptr::null_mut(),
            )?
        };

        if ret != 0 {
            return Err(ret.into());
        }

        Ok(outputs)
    }

    pub const NPU_CORE_0: u32 = RKNN_NPU_CORE_0;
    pub const NPU_CORE_1: u32 = RKNN_NPU_CORE_1;
    pub const NPU_CORE_2: u32 = RKNN_NPU_CORE_2;
    pub const NPU_CORE_ALL: u32 = RKNN_NPU_CORE_ALL;
    pub const NPU_CORE_0_1: u32 = RKNN_NPU_CORE_0_1;
    pub const NPU_CORE_0_1_2: u32 = RKNN_NPU_CORE_0_1_2;
    pub const NPU_CORE_AUTO: u32 = RKNN_NPU_CORE_AUTO;

    #[cfg(feature = "rk3576")]
    #[cfg_attr(feature = "docs", doc(cfg(feature = "rk3576")))]
    pub fn set_core_mask(&self, mask: u32) -> Result<(), Error> {
        unsafe {
            self.api.set_core_mask(self.ctx, mask)?;
        }

        Ok(())
    }
}

impl<A: RKNNAPI> Drop for RKNN<A> {
    fn drop(&mut self) {
        unsafe {
            self.api.destroy(self.ctx).unwrap();
        }
    }
}
