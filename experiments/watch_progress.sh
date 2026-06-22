#!/bin/bash
# Watch training progress for all 4 GORS experiments
watch -n 10 '
echo "===== GORS Training Progress $(date +%H:%M:%S) ====="
echo ""
for tag in gors_full gors_no_sym gors_no_phy gors_no_fb; do
    log="logs/${tag}.log"
    if [ -f "$log" ]; then
        # Get latest epoch line
        last_epoch=$(grep "Epoch" "$log" | tail -1 | grep -oP "Epoch\s+\d+" | tr -d "Epoch " | xargs)
        last_val=$(grep "Val" "$log" | tail -1 | grep -oP "RMSE=\d+\.\d+" | tr -d "RMSE=" | xargs)
        last_rho=$(grep "Val" "$log" | tail -1 | grep -oP "ρ=\d+\.\d+" | tr -d "ρ=" | xargs)
        done_marker=$(grep -c "Done\." "$log")
        
        if [ "$done_marker" -gt 0 ]; then
            status="✅ DONE"
        elif [ -n "$last_epoch" ]; then
            # Progress bar: 50 epochs total
            pct=$((last_epoch * 100 / 50))
            bar=$(printf "%-${pct}s" "=" | sed "s/ /=/g" | cut -c1-$((pct/2)))
            status="[${bar}$(printf "%$((50-pct/2))s" "" | sed 's/ /-/g')] ${last_epoch}/50"
        else
            status="Loading data..."
        fi
        
        printf "%-18s %-60s RMSE=%s ρ=%s\n" "$tag" "$status" "${last_val:-N/A}" "${last_rho:-N/A}"
    else
        printf "%-18s %s\n" "$tag" "log not found"
    fi
done
echo ""
echo "Processes: $(ps aux | grep train_gors | grep -v grep | wc -l) running"
echo "Refresh: every 10s | Ctrl+C to exit"
'
