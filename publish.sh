#!/bin/bash -e

python setup.py clean
python setup.py bdist_wheel
twine check dist/*
twine upload dist/*