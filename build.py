import numpy
import os
import shutil
from distutils.command.build_ext import build_ext
from distutils.core import Distribution, Extension

from Cython.Build import cythonize

extra_compile_args=["-std=c++11", "-O3", "-mtune=native", "-march=native"]
#extra_compile_args=["-std=c++11", "-O3", "-mtune=native", "-march=native", "-mfpmath=both", "-msse", "-msse2", "-msse3", "-msse4", "-msse4.1", "-msse4.2", "-mfma"]
include_dirs = [numpy.get_include(), 'cytm/include']


def build():
    extensions = [
        Extension(
            "*",
            ["cytm/*.pyx"],
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs,
            language="c++",
        )
    ]
    ext_modules = cythonize(
        extensions,
        annotate=True,
        compiler_directives={"binding": True, "language_level": 3},
    )

    distribution = Distribution({"name": "extended", "ext_modules": ext_modules})
    distribution.package_dir = "extended"

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)


if __name__ == "__main__":
    build()
