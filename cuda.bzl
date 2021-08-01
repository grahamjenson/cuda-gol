def cuda_default_copts():
    """Default options for all CUDA compilations."""
    return [
        "-x",
        "cuda",
        "-DGOOGLE_CUDA=1",
        "-Xcuda-fatbinary=--compress-all",
        "--no-cuda-include-ptx=all",
    ]


def cuda_library(copts=[], **kwargs):
    """Wrapper over cc_library which adds default CUDA options."""
    native.cc_library(copts=cuda_default_copts() + copts, **kwargs)
