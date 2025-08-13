use rknpu2_sys::rknn_output;

use crate::io::buffer::{BufMutView, RknnBuffer};

pub enum OutputKind<'a> {
    /// The user provides a buffer for RKNN to write into.
    Preallocated {
        buf: BufMutView<'a>,
        want_float: bool,
    },
    /// The user wants RKNN to allocate and manage the buffer.
    RuntimePreallocated { want_float: bool },
    RuntimeAllocated {
        buf: RknnBuffer,
        want_float: bool, // Necessary if there is the unlikely re-run of the same output
    },
}

pub struct Output<'a> {
    pub index: u32,
    pub kind: OutputKind<'a>,
}

impl<'a> Output<'a> {
    /// PRE-FFI-CALL: Converts the safe struct into a raw C struct.
    pub(crate) fn as_sys_output(&mut self) -> rknn_output {
        let (want_float, is_prealloc, buf, size) = match &mut self.kind {
            OutputKind::Preallocated { buf, want_float } => {
                (*want_float, 1, buf.as_mut_ptr(), buf.num_bytes())
            }
            OutputKind::RuntimePreallocated { want_float } => {
                (*want_float, 0, std::ptr::null_mut(), 0)
            }
            OutputKind::RuntimeAllocated { buf, want_float } => {
                // This case should ideally not be re-run, but if it is,
                // treat it as a pre-allocated buffer.
                (*want_float, 1, buf.ptr, buf.size)
            }
        };

        rknn_output {
            index: self.index,
            want_float: want_float as u8,
            is_prealloc,
            buf,
            size: size as u32,
        }
    }

    /// POST-FFI-CALL: Updates the safe struct from the raw C struct.
    pub(crate) fn from_sys_output(&mut self, sys_out: &rknn_output) {
        if let Some((buffer, want_float)) =
            if let OutputKind::RuntimePreallocated { want_float } = &self.kind {
                Some((
                    RknnBuffer::new(sys_out.buf, sys_out.size as usize),
                    *want_float,
                ))
            } else {
                None
            }
        {
            self.kind = OutputKind::RuntimeAllocated {
                buf: buffer,
                want_float,
            }
        }
    }
}
