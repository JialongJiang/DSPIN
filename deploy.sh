#!/usr/bin/env bash
# Clean dist, build package, upload via Twine.

PROJECT_DIR="/mnt/d/Projects/20241021_dspin_development/DSPIN"
DIST_DIR="$PROJECT_DIR/dist"
REPOSITORY="pypi"

rm -rf "$DIST_DIR"

cd "$PROJECT_DIR"
python setup.py sdist bdist_wheel

twine upload dist/* -u __token__

echo "Done."