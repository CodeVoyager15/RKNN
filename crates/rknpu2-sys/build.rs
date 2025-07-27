fn main() {
    #[cfg(all(feature = "rk35xx", not(feature = "docs"), not(feature = "libloading")))]
    println!("cargo:rustc-link-lib=rknnrt");

    #[cfg(all(feature = "rk3576", not(feature = "docs"), not(feature = "libloading")))]
    println!("cargo:rustc-link-lib=rknnrt");

    #[cfg(all(feature = "rv110x", not(feature = "docs"), not(feature = "libloading")))]
    println!("cargo:rustc-link-lib=rknnmrt");

    #[cfg(all(feature = "rk2118", not(feature = "docs"), not(feature = "libloading")))]
    println!("cargo:rustc-link-lib=rknnrt");
}
