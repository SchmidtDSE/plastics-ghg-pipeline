# Simple script to run the luigi pipeline with a single worker
#
# License: BSD, see LICENSE.md

python -m luigi --module preprocess PreprocessDataTask --local-scheduler --workers 1
