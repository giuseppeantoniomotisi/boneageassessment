#!/bin/bash

# Script file for Sphinx documentation

# Setting variables
SOURCEDIR=.
BUILDDIR=_build

# Checking if sphinx-build is installed
if ! command -v sphinx-build &> /dev/null
then
    echo "The 'sphinx-build' command was not found. Make sure you have Sphinx installed."
    echo "If you don't have Sphinx installed, grab it from https://www.sphinx-doc.org/"
    exit 1
fi

# Checking for command-line arguments
if [ -z "$1" ]; then
    # If no argument is provided, show help
    sphinx-build -M help $SOURCEDIR $BUILDDIR $SPHINXOPTS $O
else
    # Otherwise, run the specified command
    sphinx-build -M $1 $SOURCEDIR $BUILDDIR $SPHINXOPTS $O
fi
