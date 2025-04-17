less tmp1 |awk 'BEGIN{ORS=","}{print $0}' | sed 's/,$//' > tmp3
less tmp2 |awk 'BEGIN{ORS=","}{print $0}' | sed 's/,$//' > tmp4
