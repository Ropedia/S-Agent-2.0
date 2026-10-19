#!/usr/bin/env bash
# Reclaim docker space without disturbing a running experiment.
#
# SAgent builds one image per task directory, so a 400-task sweep leaves hundreds of
# images behind. On this host `docker system df` already reports ~1.4 TB of images.
#
# Only prune what is provably unreferenced: dangling layers and exited containers.
# Never run `docker system prune -a` while a sweep is live — that deletes images the
# in-flight trials still need and the run dies mid-sweep.
set -uo pipefail

echo "before: $(df -h / | awk 'NR==2{print $4" free, "$5" used"}')"
echo "running erqa containers (preserved): $(docker ps -q --filter name=erqa | wc -l)"

docker container prune -f | tail -1
docker image prune -f | tail -1

echo "after:  $(df -h / | awk 'NR==2{print $4" free, "$5" used"}')"
