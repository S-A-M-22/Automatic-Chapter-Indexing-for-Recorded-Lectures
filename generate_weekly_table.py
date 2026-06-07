import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import numpy as np

def time_to_seconds(t):
    parts = t.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(t)

def load_chapters(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chapters = []
    for ch in data:
        start = time_to_seconds(ch['start_time'])
        end = time_to_seconds(ch['end_time'])
        duration_min = (end - start) / 60.0
        key_val = str(ch.get('key', 'False')).strip()
        is_key = key_val in ('True', 'true', 'TRUE', '1', 'Yes', 'yes')
        desc = ch.get('description', '')
        word_count = len(desc.split()) if desc else 0
        chapters.append({
            'duration_min': duration_min,
            'is_key': is_key,
            'word_count': word_count
        })
    return chapters

# Define video ordering and file mappings
sw_base = r'C:\Users\amitt\Downloads\THESIS_FINAL_CODEBASE\Output\sw_visual'
tw_base = r'C:\Users\amitt\Downloads\THESIS_FINAL_CODEBASE\Output\tumbling_window'

videos = [
    ('W01', 'Week_01_-_Systems_Pr-s1-full-chapters.json', 'Week_01_-_Systems_Pr-s1-tm-chapters.json'),
    ('W02', 'Week_02_-_Systems_Pr-s1-full-chapters.json', 'Week_02_-_Systems_Pr-s1-tm-chapters.json'),
    ('W03', 'Week_03_-_Systems_Pr-s1-full-chapters.json', 'Week_03_-_Systems_Pr-s1-tm-chapters.json'),
    ('W04', 'Week_04_-_Systems_Pr-s1-full-chapters.json', 'Week_04_-_Systems_Pr-s1-tm-chapters.json'),
    ('W05', 'Week_05_-_Systems_Pr-s1-full-chapters.json', 'Week_05_-_Systems_Pr-s1-tm-chapters.json'),
    ('W06', 'Week_06_-_Systems_Pr-s1-full-chapters.json', 'Week_06_-_Systems_Pr-s1-tm-chapters.json'),
    ('W07', 'Week_07_-_Systems_Pr-s1-full-chapters.json', 'Week_07_-_Systems_Pr-s1-tm-chapters.json'),
    ('W08p1', 'sysprog_2025s1w8_1_I-s1-full-full-chapters.json', 'sysprog_2025s1w8_1_I-s1-tm-chapters.json'),
    ('W08p2', 'sysprog_2025s1w8_2_I-s1-full-full-chapters.json', 'sysprog_2025s1w8_2_I-s1-tm-chapters.json'),
    ('W08p3', 'sysprog_2025s1w8_3_I-s1-full-full-chapters.json', 'sysprog_2025s1w8_3_I-s1-full-tm-chapters.json'),
    ('W08p4', 'sysprog_2025s1w8_4_I-s1-full-full-chapters.json', 'sysprog_2025s1w8_4_I-s1-full-tm-chapters.json'),
    ('W09', 'Week_09_-_Systems_Pr-s1-full-full-chapters.json', 'Week_09_-_Systems_Pr-s1-full-tm-chapters.json'),
    ('W10', 'Week_10_-_Systems_Pr-s1-full-full-chapters.json', 'Week_10_-_Systems_Pr-s1-full-tm-chapters.json'),
    ('W11', 'Week_11_-_Systems_Pr-s1-full-full-chapters.json', 'Week_11_-_Systems_Pr-s1-full-tm-chapters.json'),
    ('W12', 'Week_12_-_Systems_Pr-s1-full-full-chapters.json', 'Week_12_-_Systems_Pr-s1-full-tm-chapters.json'),
    ('W13', 'Week_13_-_Systems_Pr-s1-full-full-chapters.json', 'Week_13_-_Systems_Pr-s1-full-tm-chapters.json'),
]

# Also try alternate SW naming for W08 parts
sw_alt = {
    'sysprog_2025s1w8_3_I-s1-full-full-chapters.json': 'sysprog_2025s1w8_3_I-s1-full-chapters.json',
    'sysprog_2025s1w8_4_I-s1-full-full-chapters.json': 'sysprog_2025s1w8_4_I-s1-full-chapters.json',
}

rows = []
all_sw_chapters = []
all_tw_chapters = []

for label, sw_file, tw_file in videos:
    sw_path = os.path.join(sw_base, sw_file)
    if not os.path.exists(sw_path) and sw_file in sw_alt:
        sw_path = os.path.join(sw_base, sw_alt[sw_file])
    tw_path = os.path.join(tw_base, tw_file)

    if not os.path.exists(sw_path):
        print(f"WARNING: SW file not found for {label}: {sw_path}")
        continue
    if not os.path.exists(tw_path):
        print(f"WARNING: TW file not found for {label}: {tw_path}")
        continue

    sw_chs = load_chapters(sw_path)
    tw_chs = load_chapters(tw_path)
    all_sw_chapters.extend(sw_chs)
    all_tw_chapters.extend(tw_chs)

    sw_total = len(sw_chs)
    sw_key = sum(1 for c in sw_chs if c['is_key'])
    sw_skip = sw_total - sw_key
    sw_key_rate = (sw_key / sw_total * 100) if sw_total else 0
    sw_avg_dur = np.mean([c['duration_min'] for c in sw_chs]) if sw_chs else 0
    sw_avg_words = np.mean([c['word_count'] for c in sw_chs]) if sw_chs else 0

    tw_total = len(tw_chs)
    tw_key = sum(1 for c in tw_chs if c['is_key'])
    tw_skip = tw_total - tw_key
    tw_key_rate = (tw_key / tw_total * 100) if tw_total else 0
    tw_avg_dur = np.mean([c['duration_min'] for c in tw_chs]) if tw_chs else 0
    tw_avg_words = np.mean([c['word_count'] for c in tw_chs]) if tw_chs else 0

    rows.append({
        'label': label,
        'sw_total': sw_total, 'sw_key': sw_key, 'sw_skip': sw_skip,
        'sw_key_rate': sw_key_rate, 'sw_avg_dur': sw_avg_dur, 'sw_avg_words': sw_avg_words,
        'tw_total': tw_total, 'tw_key': tw_key, 'tw_skip': tw_skip,
        'tw_key_rate': tw_key_rate, 'tw_avg_dur': tw_avg_dur, 'tw_avg_words': tw_avg_words,
    })

# Compute aggregate
agg_sw_total = sum(r['sw_total'] for r in rows)
agg_sw_key = sum(r['sw_key'] for r in rows)
agg_sw_skip = agg_sw_total - agg_sw_key
agg_sw_key_rate = (agg_sw_key / agg_sw_total * 100) if agg_sw_total else 0
agg_sw_avg_dur = np.mean([c['duration_min'] for c in all_sw_chapters]) if all_sw_chapters else 0
agg_sw_avg_words = np.mean([c['word_count'] for c in all_sw_chapters]) if all_sw_chapters else 0

agg_tw_total = sum(r['tw_total'] for r in rows)
agg_tw_key = sum(r['tw_key'] for r in rows)
agg_tw_skip = agg_tw_total - agg_tw_key
agg_tw_key_rate = (agg_tw_key / agg_tw_total * 100) if agg_tw_total else 0
agg_tw_avg_dur = np.mean([c['duration_min'] for c in all_tw_chapters]) if all_tw_chapters else 0
agg_tw_avg_words = np.mean([c['word_count'] for c in all_tw_chapters]) if all_tw_chapters else 0

# Print data for verification
print(f"{'Video':<8} | {'SW Tot':>6} {'SW K/S':>8} {'SW K%':>6} {'SW Dur':>8} {'SW Wds':>7} | {'TW Tot':>6} {'TW K/S':>8} {'TW K%':>6} {'TW Dur':>8} {'TW Wds':>7}")
print('-' * 100)
for r in rows:
    print(f"{r['label']:<8} | {r['sw_total']:>6} {r['sw_key']:>3}/{r['sw_skip']:<3} {r['sw_key_rate']:>5.1f}% {r['sw_avg_dur']:>7.1f}m {r['sw_avg_words']:>6.0f}w | "
          f"{r['tw_total']:>6} {r['tw_key']:>3}/{r['tw_skip']:<3} {r['tw_key_rate']:>5.1f}% {r['tw_avg_dur']:>7.1f}m {r['tw_avg_words']:>6.0f}w")
print('-' * 100)
print(f"{'Total':<8} | {agg_sw_total:>6} {agg_sw_key:>3}/{agg_sw_skip:<3} {agg_sw_key_rate:>5.1f}% {agg_sw_avg_dur:>7.1f}m {agg_sw_avg_words:>6.0f}w | "
      f"{agg_tw_total:>6} {agg_tw_key:>3}/{agg_tw_skip:<3} {agg_tw_key_rate:>5.1f}% {agg_tw_avg_dur:>7.1f}m {agg_tw_avg_words:>6.0f}w")

# ==================== GENERATE PNG TABLE ====================
n_data_rows = len(rows)
n_rows = n_data_rows + 4  # super-header + sub-header + data rows + aggregate + padding
n_cols = 11  # Week + 5 SW cols + 5 TW cols

col_widths = [1.4, 1.2, 1.3, 1.1, 1.5, 1.2, 1.2, 1.3, 1.1, 1.5, 1.2]
total_w = sum(col_widths)
fig_w = total_w + 0.4
row_h = 0.50
fig_h = (n_data_rows + 3) * row_h + 0.6

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, total_w)
ax.set_ylim(0, (n_data_rows + 3) * row_h)
ax.axis('off')
fig.patch.set_facecolor('white')

# Column x positions
col_x = [0]
for w in col_widths:
    col_x.append(col_x[-1] + w)

# Colors
sw_header_bg = '#5B8C3E'
sw_subheader_bg = '#8FBC5A'
sw_row_even = '#E8F0DC'
sw_row_odd = '#F2F7EC'
tw_header_bg = '#0D47A1'
tw_subheader_bg = '#1976D2'
tw_row_even = '#D6EAFF'
tw_row_odd = '#E8F2FF'
week_header_bg = '#4A4A4A'
week_subheader_bg = '#6A6A6A'
week_row_even = '#F0F0F0'
week_row_odd = '#FAFAFA'
agg_bg_week = '#E0E0E0'
agg_bg_sw = '#C5DC9F'
agg_bg_tw = '#BBDEFB'
border_color = '#CCCCCC'

def draw_cell(x, y, w, h, bg, text, fontsize=10, bold=False, color='#333333', ha='center'):
    rect = plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor=border_color, linewidth=0.5)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    if ha == 'center':
        tx = x + w / 2
    elif ha == 'left':
        tx = x + 0.08
    else:
        tx = x + w - 0.08
    ax.text(tx, y + h / 2, text, ha=ha, va='center', fontsize=fontsize, fontweight=weight, color=color, family='sans-serif')

# Row y positions (top to bottom: super-header, sub-header, data..., aggregate)
top_y = (n_data_rows + 2) * row_h

# === SUPER HEADER ROW ===
y = top_y
draw_cell(col_x[0], y, col_widths[0], row_h, week_header_bg, '', fontsize=11, bold=True, color='white')
sw_w = sum(col_widths[1:6])
draw_cell(col_x[1], y, sw_w, row_h, sw_header_bg, 'Sliding-Window Visual Pipeline', fontsize=11, bold=True, color='white')
tw_w = sum(col_widths[6:11])
draw_cell(col_x[6], y, tw_w, row_h, tw_header_bg, 'Tumbling-Window Pipeline', fontsize=11, bold=True, color='white')

# === SUB HEADER ROW ===
y = top_y - row_h
sub_labels = ['Week', 'Chapters', 'Key / Skip', 'Key Rate', 'Avg Duration', 'Avg Words',
              'Chapters', 'Key / Skip', 'Key Rate', 'Avg Duration', 'Avg Words']
sub_bgs = [week_subheader_bg] + [sw_subheader_bg]*5 + [tw_subheader_bg]*5
for i, (lbl, bg) in enumerate(zip(sub_labels, sub_bgs)):
    draw_cell(col_x[i], y, col_widths[i], row_h, bg, lbl, fontsize=10, bold=True, color='white')

# === DATA ROWS ===
for ri, r in enumerate(rows):
    y = top_y - (ri + 2) * row_h
    is_even = (ri % 2 == 0)
    wbg = week_row_even if is_even else week_row_odd
    sbg = sw_row_even if is_even else sw_row_odd
    tbg = tw_row_even if is_even else tw_row_odd

    draw_cell(col_x[0], y, col_widths[0], row_h, wbg, r['label'], fontsize=8.5, bold=True, ha='center')
    draw_cell(col_x[1], y, col_widths[1], row_h, sbg, str(r['sw_total']), fontsize=8.5)
    draw_cell(col_x[2], y, col_widths[2], row_h, sbg, f"{r['sw_key']} / {r['sw_skip']}", fontsize=8.5)
    draw_cell(col_x[3], y, col_widths[3], row_h, sbg, f"{r['sw_key_rate']:.0f}%", fontsize=8.5)
    draw_cell(col_x[4], y, col_widths[4], row_h, sbg, f"{r['sw_avg_dur']:.1f} min", fontsize=8.5)
    draw_cell(col_x[5], y, col_widths[5], row_h, sbg, f"{r['sw_avg_words']:.0f} wds", fontsize=8.5)
    draw_cell(col_x[6], y, col_widths[6], row_h, tbg, str(r['tw_total']), fontsize=8.5)
    draw_cell(col_x[7], y, col_widths[7], row_h, tbg, f"{r['tw_key']} / {r['tw_skip']}", fontsize=8.5)
    draw_cell(col_x[8], y, col_widths[8], row_h, tbg, f"{r['tw_key_rate']:.0f}%", fontsize=8.5)
    draw_cell(col_x[9], y, col_widths[9], row_h, tbg, f"{r['tw_avg_dur']:.1f} min", fontsize=8.5)
    draw_cell(col_x[10], y, col_widths[10], row_h, tbg, f"{r['tw_avg_words']:.0f} wds", fontsize=8.5)

# === AGGREGATE ROW ===
y = top_y - (n_data_rows + 2) * row_h
draw_cell(col_x[0], y, col_widths[0], row_h, agg_bg_week, 'Total', fontsize=11, bold=True, ha='center')
draw_cell(col_x[1], y, col_widths[1], row_h, agg_bg_sw, str(agg_sw_total), fontsize=8.5, bold=True)
draw_cell(col_x[2], y, col_widths[2], row_h, agg_bg_sw, f"{agg_sw_key} / {agg_sw_skip}", fontsize=8.5, bold=True)
draw_cell(col_x[3], y, col_widths[3], row_h, agg_bg_sw, f"{agg_sw_key_rate:.1f}%", fontsize=8.5, bold=True)
draw_cell(col_x[4], y, col_widths[4], row_h, agg_bg_sw, f"{agg_sw_avg_dur:.1f} min", fontsize=8.5, bold=True)
draw_cell(col_x[5], y, col_widths[5], row_h, agg_bg_sw, f"{agg_sw_avg_words:.0f} wds", fontsize=8.5, bold=True)
draw_cell(col_x[6], y, col_widths[6], row_h, agg_bg_tw, str(agg_tw_total), fontsize=8.5, bold=True)
draw_cell(col_x[7], y, col_widths[7], row_h, agg_bg_tw, f"{agg_tw_key} / {agg_tw_skip}", fontsize=8.5, bold=True)
draw_cell(col_x[8], y, col_widths[8], row_h, agg_bg_tw, f"{agg_tw_key_rate:.1f}%", fontsize=8.5, bold=True)
draw_cell(col_x[9], y, col_widths[9], row_h, agg_bg_tw, f"{agg_tw_avg_dur:.1f} min", fontsize=8.5, bold=True)
draw_cell(col_x[10], y, col_widths[10], row_h, agg_bg_tw, f"{agg_tw_avg_words:.0f} wds", fontsize=8.5, bold=True)

out_path = r'C:\Users\amitt\Downloads\Improving_Lecture_Engagement_with_Automatic_Chapter_Indexing\figures\weekly-chap-stat.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.05, facecolor='white')
plt.close()
print(f"\nSaved to: {out_path}")
