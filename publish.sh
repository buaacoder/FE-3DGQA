#!/bin/bash -e

python setup.py clean
python setup.py bdist_wheel --plat-name=manylinux1_x86_64
twine check dist/*
twine upload dist/*