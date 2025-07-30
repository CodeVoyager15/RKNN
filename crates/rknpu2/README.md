# rknpu2

## High level bindings for [RKNN-Toolkit2 C inference API](https://github.com/airockchip/rknn-toolkit2)

## crate features
- rv110x # For RV1103 / RV1106 / B variants
- rk2118 # For RK2118
- rk35xx # For RK356x
- rk3576 # For RK3576 / RK3588
- libloading
- docs

The rk3576, rk35xx, rk2118, rv110x features determines what library to link with (librknnrt.so or librknnmrt.so)

The libloading feature enables dynamic loading of the RKNN-Toolkit2 library at runtime.
