#!/bin/zsh
# Wait until every submission whose description contains one of the given substrings has a public score; print them.
cd /Users/shreyanshgoyal/Documents/compi
T() { perl -e 'alarm 150; exec @ARGV' "$@"; }
while true; do
  csv=$(T .venv/bin/kaggle competitions submissions rsna-knee-abnormality-detection --csv 2>/dev/null)
  done_all=1; out=""
  for key in "$@"; do
    row=$(echo "$csv" | grep -F -- "$key" | head -1)
    if [ -z "$row" ] || echo "$row" | grep -q PENDING; then done_all=0; else out="$out\n$key => $(echo "$row" | awk -F, '{print $(NF-1)}')"; fi
  done
  [ $done_all -eq 1 ] && { echo "$(date -u +%H:%M)$out"; exit 0; }
  sleep 600
done
