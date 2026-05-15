"""
Setup para compilação Cython

Compilar com:
  python setup.py build_ext --inplace

Após compilação, um arquivo regex_cython.pyd (Windows) ou regex_cython.so (Linux)
será criado no mesmo diretório, permitindo:
  from regex_cython import tipo_documento, valor_danos_morais, processar_batch, ...
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    name="regex_cython",
    ext_modules=cythonize(
        "regex_cython.pyx",
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
        annotate=True,  # Gera HTML com análise de performance
    ),
    zip_safe=False,
)
