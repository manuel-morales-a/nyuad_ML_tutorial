#!/usr/bin/env bash
# Download the Top Quark Tagging Reference Dataset (https://zenodo.org/records/2603256).
# Resumable: re-run it if the connection drops. ~1.7 GB in total.
set -euo pipefail
DIR="${1:-Top_Quark_Tagging_dataset}"
mkdir -p "$DIR"
for f in train val test; do
  url="https://zenodo.org/records/2603256/files/$f.h5"
  until curl -L --fail -C - --retry 5 --speed-limit 10000 --speed-time 60 -o "$DIR/$f.h5" "$url"; do
    echo "Download of $f.h5 interrupted, resuming..." >&2
    sleep 5
  done
done
ls -lh "$DIR"
