import os
import platform
import sys
from setuptools import setup
import pybind11
from torch.utils import cpp_extension

__version__ = "0.1.0"

# JPEG header/library pair. Override with JPEG_INCLUDE_DIR / JPEG_LIB_DIR; otherwise common
# locations (conda env, Debian/Ubuntu multiarch, RHEL/HPC lib64) are probed in order.
# The pair is kept fully explicit so Python/sysroot include paths do not silently pull in a
# different `jconfig.h`, and the library dir is added to the rpath so the same libjpeg is
# resolved at run time.
def _candidates(env_name, defaults):
    value = os.environ.get(env_name)
    return [value] if value else defaults


_arch = platform.machine()
_prefixes = [p for p in (os.environ.get("CONDA_PREFIX"), sys.prefix) if p]
_include_candidates = _candidates(
    "JPEG_INCLUDE_DIR",
    [os.path.join(p, "include") for p in _prefixes]
    + [f"/usr/include/{_arch}-linux-gnu", "/usr/include", "/usr/local/include"],
)
_lib_candidates = _candidates(
    "JPEG_LIB_DIR",
    [os.path.join(p, "lib") for p in _prefixes]
    + [f"/usr/lib/{_arch}-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/local/lib"],
)


def _first_with(dirs, filename):
    for d in dirs:
        if os.path.isfile(os.path.join(d, filename)):
            return d
    raise RuntimeError(
        f"Cannot find {filename} in any of {dirs}. "
        "Install libjpeg development files (e.g. `apt install libjpeg-dev` or "
        "`conda install libjpeg-turbo`) or set JPEG_INCLUDE_DIR / JPEG_LIB_DIR."
    )


jpeg_include_dir = _first_with(_include_candidates, "jpeglib.h")
jpeg_library_dir = _first_with(_lib_candidates, "libjpeg.so")
jpeg_library_file = os.path.join(jpeg_library_dir, "libjpeg.so")
jpeg_header = os.path.join(jpeg_include_dir, "jpeglib.h")

include_dirs = [jpeg_include_dir]
include_dirs.append(pybind11.get_include())
library_dirs = [jpeg_library_dir]
libraries = []
extra_objects = [jpeg_library_file]
headers = [jpeg_header]
extra_compile_args = ["-std=c++17"]
extra_link_args = [f"-Wl,-rpath,{jpeg_library_dir}"]

ext_modules = [
    cpp_extension.CppExtension(
        "dct_manip",
        ["dct_manip.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_objects=extra_objects,
        depends=headers,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    cpp_extension.CppExtension(
        "byte_manip",
        ["byte_manip.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_objects=extra_objects,
        depends=headers,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="jpeg_manip_ext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    version=__version__,
)
