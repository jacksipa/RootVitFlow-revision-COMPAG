#!/bin/bash
# Sale (y avisa) cuando aparece un error nuevo en los logs, cuando el watchdog muere o cuando todo termina.
cd "$(dirname "$0")"
count() { cat logs/queue*.log logs/seg_*.log logs/xai.log logs/summarize.log logs/organize.log 2>/dev/null | grep -cE "Traceback|OutOfMemory|Killed"; }
base=$(count)
while true; do
  [ -f logs/ALL_DONE ] && { echo "ALL_DONE"; cat logs/STATUS.txt; exit 0; }
  pgrep -f "bash run_all.sh" > /dev/null || { echo "WATCHDOG_MUERTO"; tail -5 logs/run_all.log; exit 2; }
  now=$(count)
  if [ "$now" -gt "$base" ]; then echo "NUEVO_ERROR ($base -> $now)"; grep -lE "Traceback|OutOfMemory" logs/*.log | xargs ls -t | head -3; exit 1; fi
  sleep 60
done
