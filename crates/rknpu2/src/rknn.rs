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
}

impl Drop for RKNN {
    fn drop(&mut self) {
        unsafe {
            rknpu2_sys::rknn_destroy(self.ctx);
        }
    }
}
