#!/usr/bin/env bash
#
# quick_line_shard.sh  —  Slice a text file into N shards with ≈equal lines
#
#   ./quick_line_shard.sh INPUT_FILE N  [PREFIX]
#     • INPUT_FILE : path to the text file to split
#     • N          : number of shards you want (positive integer)
#     • PREFIX     : optional prefix for output files (default: INPUT_FILE)
#
#   Output files are named   PREFIX.part00, PREFIX.part01, …

set -euo pipefail

FILE=${1:?Need an input file}
SHARDS=${2:?Need a shard count}
PREFIX=${3:-$FILE}

[[ -r $FILE ]]                                 || { echo "❌  Cannot read $FILE" >&2; exit 2; }
[[ $SHARDS =~ ^[0-9]+$ && $SHARDS -gt 0 ]]     || { echo "❌  N must be a positive integer" >&2; exit 2; }

TOTAL=$(wc -l < "$FILE")                       # total number of lines
BASE=$(( TOTAL / SHARDS ))                     # lines every shard is guaranteed
EXTRA=$(( TOTAL % SHARDS ))                    # first EXTRA shards get one extra line
PADW=${#SHARDS}; (( PADW < 2 )) && PADW=2      # zero‑padding width for part numbers

echo "Total lines : $TOTAL"
echo "Shards       : $SHARDS  ( $BASE each, +1 line to first $EXTRA shards )"
echo

start=1
for i in $(seq 0 $(( SHARDS - 1 ))); do
  count=$BASE
  (( i < EXTRA )) && count=$(( count + 1 ))     # add the remainder line if still owed

  end=$(( start + count - 1 ))
  part="${PREFIX}.part$(printf "%0${PADW}d" "$i")"

  # sed streams only the slice we need → efficient, keeps order
  sed -n "${start},${end}p" "$FILE" > "$part"

  printf "▶  %-20s  %6d lines  (rows %d‑%d)\n" "$part" "$count" "$start" "$end"
  start=$(( end + 1 ))
done
