"""
Maternal Fitband — Session Analyzer
=====================================
Run this AFTER a recording session to get a statistical summary
of all HR and SpO2 readings collected in readings.csv.

Usage:
  python data/analyze.py
  python data/analyze.py --csv data/readings.csv
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Maternal Fitband Session Analyzer')
parser.add_argument('--csv', default=os.path.join(os.path.dirname(__file__), 'readings.csv'),
                    help='Path to CSV file from serial_reader.py')
args = parser.parse_args()

if not os.path.exists(args.csv):
    print(f"[ERR] CSV not found: {args.csv}")
    print("      Run serial_reader.py first to collect data.")
    sys.exit(1)

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(args.csv)
if df.empty:
    print("[ERR] CSV file is empty. No data to analyze.")
    sys.exit(1)

df['wall_time'] = pd.to_datetime(df['wall_time'])

# Convert boolean string columns if needed
for col in ['hr_valid', 'spo2_valid']:
    if df[col].dtype == object:
        df[col] = df[col].map({'True': True, 'False': False, True: True, False: False})

df_valid = df[(df['hr_valid'] == True) & (df['spo2_valid'] == True)].copy()
df_hr    = df[df['hr_valid']   == True].copy()
df_sp    = df[df['spo2_valid'] == True].copy()

# ── Console Summary ────────────────────────────────────────────────────────────
sep = "=" * 60
print(f"\n{sep}")
print("  MATERNAL FITBAND -- SESSION ANALYSIS REPORT")
print(sep)
print(f"  CSV file     : {args.csv}")
print(f"  Session start: {df['wall_time'].min()}")
print(f"  Session end  : {df['wall_time'].max()}")
duration = (df['wall_time'].max() - df['wall_time'].min()).total_seconds() / 60
print(f"  Duration     : {duration:.1f} minutes")
print(f"  Total epochs : {len(df)}")
print(f"  Valid HR     : {len(df_hr)} ({100*len(df_hr)/len(df):.0f}%)")
print(f"  Valid SpO2   : {len(df_sp)} ({100*len(df_sp)/len(df):.0f}%)")
print(sep)

if len(df_hr) > 0:
    print(f"\n  HEART RATE (bpm)")
    print(f"    Mean    : {df_hr['hr_bpm'].mean():.1f}")
    print(f"    Std Dev : {df_hr['hr_bpm'].std():.1f}")
    print(f"    Min     : {df_hr['hr_bpm'].min():.0f}")
    print(f"    Max     : {df_hr['hr_bpm'].max():.0f}")
    tachycardia = df_hr[df_hr['hr_bpm'] > 100]
    bradycardia = df_hr[df_hr['hr_bpm'] < 50]
    if not tachycardia.empty:
        print(f"\n  [!] Tachycardia (HR > 100) detected in {len(tachycardia)} epoch(s)")
    if not bradycardia.empty:
        print(f"  [!] Bradycardia (HR < 50) detected in {len(bradycardia)} epoch(s)")

if len(df_sp) > 0:
    print(f"\n  BLOOD OXYGEN / SpO2 (%)")
    print(f"    Mean    : {df_sp['spo2_pct'].mean():.1f}")
    print(f"    Std Dev : {df_sp['spo2_pct'].std():.1f}")
    print(f"    Min     : {df_sp['spo2_pct'].min():.0f}")
    print(f"    Max     : {df_sp['spo2_pct'].max():.0f}")
    low_spo2 = df_sp[df_sp['spo2_pct'] < 94]
    caution  = df_sp[(df_sp['spo2_pct'] >= 94) & (df_sp['spo2_pct'] < 95)]
    if not low_spo2.empty:
        print(f"\n  [CRITICAL] SpO2 < 94% detected in {len(low_spo2)} epoch(s):")
        print(low_spo2[['wall_time', 'hr_bpm', 'spo2_pct']].to_string(index=False))
    elif not caution.empty:
        print(f"\n  [CAUTION] SpO2 between 94-95% in {len(caution)} epoch(s)")
    else:
        print(f"\n  [OK] No SpO2 dips below 94%.")

if len(df) > 0:
    print(f"\n  ACTIVITY DISTRIBUTION")
    act_counts = df['activity'].value_counts()
    for act, count in act_counts.items():
        bar = '#' * int(30 * count / len(df))
        pct = 100 * count / len(df)
        print(f"    {act:<12s} {bar:<30s} {pct:4.0f}% ({count})")

print(f"\n{sep}\n")

# ── Plot ───────────────────────────────────────────────────────────────────────
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#0D1117')
fig.suptitle('Maternal Fitband -- Session Analysis Report',
             fontsize=15, fontweight='bold', color='white', y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
ax_hr   = fig.add_subplot(gs[0, :])   # HR timeline (full width)
ax_sp   = fig.add_subplot(gs[1, :])   # SpO2 timeline (full width)
ax_vm   = fig.add_subplot(gs[2, 0])   # VM timeline
ax_pie  = fig.add_subplot(gs[2, 1])   # Activity pie chart

PANEL_BG = '#161B22'
RED      = '#E63946'
BLUE     = '#457B9D'
TEAL     = '#2A9D8F'
ORANGE   = '#F4A261'
GRAY     = '#8B949E'

for ax in [ax_hr, ax_sp, ax_vm]:
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, alpha=0.15, color=GRAY)
    ax.tick_params(colors=GRAY, labelsize=8)
    for s in ['top','right']:  ax.spines[s].set_visible(False)
    for s in ['bottom','left']:ax.spines[s].set_color(GRAY)

# HR timeline
if len(df_hr) > 0:
    ax_hr.plot(df_hr['wall_time'], df_hr['hr_bpm'], color=RED, linewidth=1.5, marker='o', markersize=3)
    ax_hr.axhline(y=100, color=ORANGE, linestyle='--', alpha=0.7, linewidth=1, label='Tachycardia')
    ax_hr.axhline(y=60,  color=GRAY,   linestyle=':',  alpha=0.5, linewidth=1, label='Normal lower')
    ax_hr.fill_between(df_hr['wall_time'], df_hr['hr_bpm'], alpha=0.1, color=RED)
    ax_hr.set_ylabel('Heart Rate (bpm)', color=GRAY, fontsize=9)
    ax_hr.set_title('Heart Rate Over Session', color='white', fontsize=10)
    ax_hr.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=GRAY)
    ax_hr.set_ylim(40, 130)

# SpO2 timeline
if len(df_sp) > 0:
    ax_sp.plot(df_sp['wall_time'], df_sp['spo2_pct'], color=BLUE, linewidth=1.5, marker='s', markersize=3)
    ax_sp.axhline(y=94, color=RED,    linestyle='--', alpha=0.8, linewidth=1.5, label='Critical (<94%)')
    ax_sp.axhline(y=95, color=ORANGE, linestyle=':',  alpha=0.6, linewidth=1,   label='Caution (<95%)')
    ax_sp.fill_between(df_sp['wall_time'], df_sp['spo2_pct'], 85, alpha=0.08, color=BLUE)
    ax_sp.set_ylabel('SpO2 (%)', color=GRAY, fontsize=9)
    ax_sp.set_title('Blood Oxygen (SpO2) Over Session', color='white', fontsize=10)
    ax_sp.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=GRAY)
    ax_sp.set_ylim(85, 101)

# VM timeline
if 'vm_g' in df.columns:
    ax_vm.fill_between(df['wall_time'], df['vm_g'], alpha=0.3, color=TEAL)
    ax_vm.plot(df['wall_time'], df['vm_g'], color=TEAL, linewidth=1.2)
    ax_vm.set_ylabel('Vector Magnitude (g)', color=GRAY, fontsize=9)
    ax_vm.set_title('Activity (VM)', color='white', fontsize=10)
    ax_vm.tick_params(axis='x', rotation=25)

# Activity pie chart
if 'activity' in df.columns:
    act_counts = df['activity'].value_counts()
    colors_pie = [RED, ORANGE, TEAL, BLUE, GRAY][:len(act_counts)]
    ax_pie.pie(
        act_counts.values,
        labels=act_counts.index,
        colors=colors_pie,
        autopct='%1.0f%%',
        textprops={'color': 'white', 'fontsize': 9},
        startangle=90,
        wedgeprops={'linewidth': 0.5, 'edgecolor': '#0D1117'}
    )
    ax_pie.set_title('Activity Distribution', color='white', fontsize=10)
    ax_pie.set_facecolor(PANEL_BG)

output_path = os.path.join(os.path.dirname(args.csv), 'session_summary.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
plt.show()
print(f"[DONE] Session summary chart saved to: {output_path}\n")
