# Simple script to run the luigi pipeline with a single worker
#
# License: BSD, see LICENSE.md

python -m luigi --module goods_project ProjectionTask --local-scheduler --workers 1
