"""Pipeline diagram: the original TMD vs our defense, plus a schematic of the embedding space.
Usage: python make_pipeline_svg.py out.svg"""
import sys

W, H = 1200, 790
INK, MUTED = '#1f2933', '#52606d'
SAME = ('#eef2f7', '#6b7a90')        # unchanged component
NEW = ('#fde8d4', '#d9822b')         # changed in our defense
TRAIN = ('#ffffff', '#9aa5b4')       # training time only (dashed)
FONT = 'Helvetica, Arial, sans-serif'
out = []


def text(x, y, s, size=14, weight='normal', anchor='start', color=INK):
    s = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    out.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
               f'text-anchor="{anchor}" fill="{color}">{s}</text>')


def box(x, y, w, h, lines, colors, dashed=False, size=15):
    fill, stroke = colors
    dash = ' stroke-dasharray="6,4"' if dashed else ''
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" '
               f'stroke-width="2"{dash}/>')
    lh = size + 5
    y0 = y + h / 2 - (len(lines) - 1) * lh / 2 + size / 3
    for i, s in enumerate(lines):
        text(x + w / 2, y0 + i * lh, s, size if i == 0 else size - 2, 'bold' if i == 0 and not dashed else 'normal',
             'middle')


def arrow(x1, y1, x2, y2, label=None, color=MUTED, dashed=False):
    dash = ' stroke-dasharray="5,4"' if dashed else ''
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2"{dash} '
               f'marker-end="url(#arrow)"/>')
    if label:
        text((x1 + x2) / 2, y1 - 9, label, 13, 'normal', 'middle', MUTED)


def lane(y0, title, bert, proj, train_bert, train_proj, note, changed):
    text(20, y0 + 18, title, 18, 'bold')
    y, h = y0 + 36, 64
    xs = [(20, 120), (170, 160), (425, 120), (600, 225), (915, 120), (1070, 110)]
    boxes = [['Sentence'], bert, ['Q(c | t)', 'pick a piece'], proj, ['Classifier'], ['Label']]
    for i, ((x, w), lines) in enumerate(zip(xs, boxes)):
        box(x, y, w, h, lines, NEW if (changed and i in (1, 3)) else SAME)
    for i, lbl in enumerate([None, 'embedding t', None, 'projected t̂', None]):
        (xa, wa), (xb, _) = xs[i], xs[i + 1]
        arrow(xa + wa, y + h / 2, xb - 4, y + h / 2, lbl)
    ty = y + h + 30
    box(20, ty, 310, 44, [train_bert], TRAIN, dashed=True, size=14)
    box(425, ty, 400, 44, [train_proj], TRAIN, dashed=True, size=14)
    for x in (250, 485, 712):
        arrow(x, ty, x, y + h + 4, dashed=True, color=TRAIN[1])
    text(1070, ty + 27, note, 14, 'bold', 'middle', NEW[1] if changed else MUTED)


def band(ox, oy, p0, p1, p2, color):
    out.append(f'<path d="M {ox + p0[0]} {oy + p0[1]} Q {ox + p1[0]} {oy + p1[1]} {ox + p2[0]} {oy + p2[1]}" '
               f'fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round" opacity="0.8"/>')


def dot(x, y, color, r=5, ring=None):
    extra = f' stroke="{ring}" stroke-width="3"' if ring else ''
    out.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"{extra}/>')


def panel(ox, oy, title, caption, kind):
    text(ox + 110, oy - 12, title, 15, 'bold', 'middle')
    out.append(f'<rect x="{ox}" y="{oy}" width="220" height="150" rx="8" fill="#fbfcfd" stroke="#d5dbe3"/>')
    A, B = ((15, 95), (100, 25), (205, 55)), ((15, 135), (110, 85), (205, 125))
    band(ox, oy, *A, '#9ec5f0'); band(ox, oy, *B, '#b5e0b0')
    text(ox + 208, oy + 44, 'World', 11, 'normal', 'end', '#2f6fb0')
    text(ox + 208, oy + 146, 'Sports', 11, 'normal', 'end', '#3d8b37')
    clean = (ox + 105, oy + 50)
    dot(*clean, '#2f6fb0')
    if kind == 'attack':
        adv = (ox + 120, oy + 98)
        arrow(clean[0], clean[1] + 6, adv[0], adv[1] - 8, color='#c0392b', dashed=True)
        dot(*adv, '#c0392b')
    elif kind == 'tmd':
        adv = (ox + 118, oy + 80)
        dot(*adv, '#c0392b')
        for t in (0.08, 0.22, 0.38, 0.62, 0.78, 0.93):
            x = (1 - t) ** 2 * A[0][0] + 2 * (1 - t) * t * A[1][0] + t ** 2 * A[2][0]
            y = (1 - t) ** 2 * A[0][1] + 2 * (1 - t) * t * A[1][1] + t ** 2 * A[2][1]
            dot(ox + x, oy + y, '#8a96a8', 4, ring='#d9822b' if t == 0.62 else None)
        arrow(adv[0] + 2, adv[1] - 7, ox + 127, oy + 54, color='#d9822b', dashed=True)
    else:
        adv = (ox + 122, oy + 82)
        dot(*adv, '#c0392b')
        arrow(adv[0] - 3, adv[1] - 7, clean[0] + 4, clean[1] + 9, color='#d9822b')
    text(ox + 110, oy + 172, caption, 13, 'normal', 'middle', MUTED)


out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
out.append('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker></defs>')
out.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
text(20, 36, 'TMD vs our defense', 22, 'bold')
box(865, 14, 150, 30, ['changed in ours'], NEW, size=13)
box(1030, 14, 150, 30, ['training only'], TRAIN, dashed=True, size=13)

lane(66, 'Original TMD', ['BERT'], ['Random search', 'closest of 30 samples'],
     'trained on clean sentences', 'GAN trained on clean embeddings', 'random on every query', False)
lane(276, 'Ours', ['BERT', 'adversarially trained'], ['Learned encoder', 'one step, no guessing'],
     'also trained on attacked training sentences', 'encoder learns: attacked → clean', 'deterministic', True)

box(20, 482, 1160, 32, ['Same evaluation for both: the authors\' attack code on 1000 test sentences'], SAME, size=14)

text(20, 556, 'Embedding space (schematic)', 16, 'bold')
panel(150, 604, 'Attack', 'pushed off its band', 'attack')
panel(490, 604, 'TMD', 'closest of 30 random guesses', 'tmd')
panel(830, 604, 'Ours', 'moves less, mapped straight back', 'ours')
out.append('</svg>')
open(sys.argv[1], 'w').write('\n'.join(out))
