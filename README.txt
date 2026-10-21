Messi Valuation with Chain Credit

Why: An earlier project valued Messi by swapping his goals and assists for a replacement player’s, and found almost nothing (16 actual points against 16.7 simulated), because he scored or assisted in only 2 of 6 matches. This project redoes the swap with shared credit. It asks how much of Argentina’s chance quality Messi was part of, and what changes if an average attacker is part of it instead.

What it does: For each of the 5 Argentina matches Messi played in Copa America 2024 (he missed the Peru game), the script takes Argentina’s non-penalty shot xG and measures Messi’s share of it under three credit views: shots plus key passes (the goals plus assists view, measured in xG), xG chain, and xG buildup. It then replaces his share with one drawn at random from attackers in the tournament (226 attacker-matches from 86 players, each with at least 40 touches, zeros kept). The adjusted xG becomes expected points using Poisson goals, repeated 10,000 times per measure. A second, weaker replacement pool is run as a check.

Key findings: Messi was part of between 35 percent and 85 percent of Argentina’s shot xG per match, against 39 percent for an average attacker. Against an average attacker his value is +1.59 points under xG chain (5th to 95th percentile +0.40 to +2.86, positive in 99 percent of runs), +0.95 under shots plus key passes, and +0.61 under xG buildup. The xG version of goals plus assists is not near zero, so the earlier near-zero result came mostly from counting only finished goals and assists. In the Colombia match he was part of 35 percent of the xG, below the 39 percent average, so the swap changed nothing (-0.03 points). He was substituted in that final, which is a likely reason but was not tested. Expected points from xG alone were 10.51, against 13 real points.

Limitations: The model assumes the moves Messi was part of would not happen without him, so the values are an upper bound. The weaker-pool results are higher still. Touches stand in for minutes played because lineup data was not pulled. Every touch counts equally, and defending and off-ball movement are not credited. Five matches is a small sample.

Visual: messi_value_by_measure.png shows the points Messi adds under each credit view. messi_share_by_match.png shows his share of Argentina’s xG in each match against the average attacker. messi_swap_replay.mp4 shows the swap match by match and ends with both charts.

Tools: Python, pandas, NumPy, matplotlib, imageio-ffmpeg, statsbombpy.

Data source: StatsBomb open data, Copa America 2024. Penalties and shootouts excluded.