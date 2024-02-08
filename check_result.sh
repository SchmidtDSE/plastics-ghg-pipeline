# Simple script to ensure the confirmation file is present.
#
# License: BSD, see LICENSE.md

[ -f "deploy/confirm.txt" ] || exit 1
