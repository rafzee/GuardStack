#!/bin/bash
# Your jobs at a glance: what runs, when the waiting ones should start, how far each running one is.
#   on the cluster:      bash ~/work/guardstack/slurm/status.sh
#   from the Mac: ssh <cluster> 'bash ~/work/guardstack/slurm/status.sh'
cd "${PROJ:-$HOME/work/guardstack}" || exit 1

echo "== running"
squeue --me -t R -o "%.14i %.12j %.10M %.11l %.10L %R"
echo "   TIME = elapsed; TIME_LEFT = until SLURM would stop it at its limit (an upper bound, not an ETA)"

echo
echo "== waiting"
squeue --me -t PD --start -o "%.14i %.12j %.20S %.11l %R"
echo "   START_TIME = SLURM's current estimate (N/A while the job waits on a dependency or on your"
echo "   4-running-jobs limit; it then starts when one of your running jobs ends)"

echo
echo "== progress of running jobs (last milestone in each job's log)"
for id in $(squeue --me -t R -h -o "%i"); do
  base=${id%%_*}; task=${id#"$base"}; task=${task#_}
  if [ -n "$task" ]; then pat="logs/*_${base}_${task}."; else pat="logs/*_${base}."; fi
  last=""
  for f in ${pat}err ${pat}out; do      # the python log (.err) has the per-stack ETA lines
    [ -f "$f" ] || continue
    last=$(grep -E " ETA |stack=|ready after|^== |^started |exited|unpacked|Error" "$f" \
             | grep -v "HTTP Request" | tail -1 | cut -c1-160)
    [ -n "$last" ] && break
  done
  printf "%-14s %s\n" "$id" "${last:-(no milestone yet)}"
done
echo "   ETA lines are per stack; a job runs its stacks one after another (see STACKS in its .out)"
