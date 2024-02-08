# Simple script to run the luigi pipeline with a single worker
#
# License: BSD, see LICENSE.md

rm -r deploy
mkdir deploy
python -m luigi --module tasks SweepAndProjectTask --local-scheduler --workers 1
