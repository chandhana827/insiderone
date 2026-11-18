# Job board applications analysis (Jul-Sep 2026)
Re-run: `pip install pandas numpy matplotlib`, put job-board-applications.csv next to analysis.py, then `python analysis.py`.
Random seed fixed (42) so bootstrap intervals repeat. Console output saved in run_output.txt.

Files: note_to_manager.md (the note), chart_q2_shortlist_by_steps.png (the chart), analysis.py (all work),
applications_clean.csv (cleaned data), out_q1_*.csv (Q1 tables).

Cleaning: removed 6 exact duplicate rows (1206 -> 1200); standardised 9 work_mode labels (On-site/Hybrid/Remote);
15 blank sources labelled 'unknown' and left out of Q1 only. Logic checks (hired within interviewed within shortlisted,
dates in range) all pass. Five reply delays of 20-32 days were kept (plausible; medians used).
Method: rates = share of applications; 95% intervals by bootstrap (5,000 resamples).

