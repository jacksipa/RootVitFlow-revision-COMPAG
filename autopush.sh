#!/bin/bash
# Sube los resultados al repo privado cada 15 min; termina con un push final tras ALL_DONE.
cd "$(dirname "$0")"
push() {
  git add -A >/dev/null 2>&1
  if ! git diff --cached --quiet; then
    git commit -qm "Actualización automática de resultados ($(date '+%F %H:%M'))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin main && echo "[$(date '+%T')] push ok"
  fi
}
while true; do
  push
  if [ -f logs/ALL_DONE ]; then sleep 120; push; echo "[$(date '+%T')] push final"; exit 0; fi
  sleep 900
done
