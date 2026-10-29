"""Pipeline diagram: the original TMD vs our defense (notebook 07b), plus a schematic of the embedding space."""
import sys

W, H = 1200, 925
INK, MUTED = '#1f2933', '#52606d'
SAME = ('#eef2f7', '#6b7a90')        # unchanged component
NEW = ('#fde8d4', '#d9822b')         # changed in our defense
TRAIN = ('#ffffff', '#9aa5b4')       # training time only (dashed)
FONT = 'Helvetica, Arial, sans-serif'
out = []


def text(x, y, s, size=14, weight='normal', anchor='start', color=INK, style='normal'):
    s = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    out.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
               f'font-style="{style}" text-anchor="{anchor}" fill="{color}">{s}</text>')


def box(x, y, w, h, lines, colors, dashed=False, size=14, bold_first=True):
    fill, stroke = colors
    dash = ' stroke-dasharray="6,4"' if dashed else ''
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" '
               f'stroke-width="2"{dash}/>')
    lh = size + 5
    y0 = y + h / 2 - (len(lines) - 1) * lh / 2 + size / 3
    for i, s in enumerate(lines):
        text(x + w / 2, y0 + i * lh, s, size, 'bold' if (bold_first and i == 0) else 'normal', 'middle')


def arrow(x1, y1, x2, y2, label=None, color=MUTED, dashed=False, above=True):
    dash = ' stroke-dasharray="5,4"' if dashed else ''
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2"{dash} '
               f'marker-end="url(#arrow)"/>')
    if label:
        text((x1 + x2) / 2, y1 - 8 if above else y1 + 18, label, 12, 'normal', 'middle', MUTED)


def lane(y0, title, bert, proj, train_bert, train_proj, note, changed):
    text(20, y0 + 18, title, 18, 'bold')
    y, h = y0 + 40, 84
    xs = [(20, 110), (170, 160), (415, 140), (625, 240), (950, 130), (1110, 70)]
    boxes = [['Sentence x'], bert, ['Q(c | t)', 'which piece of the', 'manifold?'], proj,
             ['Classifier', 'BERT\'s own', 'layer'], ['Label']]
    for i, ((x, w), lines) in enumerate(zip(xs, boxes)):
        box(x, y, w, h, lines, NEW if (changed and i in (1, 3)) else SAME)
    labels = [None, 'embedding t', 'piece c', 'projected t̂', None]
    for i in range(5):
        (xa, wa), (xb, _) = xs[i], xs[i + 1]
        arrow(xa + wa, y + h / 2, xb - 4, y + h / 2, labels[i])
    ty, th = y + h + 34, 92
    box(20, ty, 310, th, train_bert, TRAIN, dashed=True, size=13)
    box(415, ty, 450, th, train_proj, TRAIN, dashed=True, size=13)
    for x in (250, 485, 745):
        arrow(x, ty, x, y + h + 4, dashed=True, color=TRAIN[1])
    text(1065, ty + th / 2 - 4, note[0], 13, 'bold', 'middle', NEW[1] if changed else MUTED)
    text(1065, ty + th / 2 + 15, note[1], 13, 'normal', 'middle', MUTED)


def band(ox, oy, p0, p1, p2, color):
    out.append(f'<path d="M {ox + p0[0]} {oy + p0[1]} Q {ox + p1[0]} {oy + p1[1]} {ox + p2[0]} {oy + p2[1]}" '
               f'fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round" opacity="0.8"/>')


def dot(x, y, color, r=5, ring=None):
    extra = f' stroke="{ring}" stroke-width="3"' if ring else ''
    out.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"{extra}/>')


def panel(ox, oy, title, kind):
    text(ox + 110, oy - 12, title, 14, 'bold', 'middle')
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
        text(ox + 128, oy + 80, 'attacked', 11, 'normal', 'start', '#c0392b')
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


out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
out.append('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker></defs>')
out.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
text(20, 34, 'The original TMD vs our defense: what changes', 22, 'bold')
box(700, 14, 150, 30, ['unchanged'], SAME, size=12, bold_first=False)
box(865, 14, 150, 30, ['changed in ours'], NEW, size=12, bold_first=False)
box(1030, 14, 150, 30, ['training time only'], TRAIN, dashed=True, size=12, bold_first=False)

lane(64, 'Original TMD (Nguyen Minh et al., EMNLP 2022)',
     ['BERT', 'fine-tuned on', 'clean sentences'],
     ['Random search', '30 random z, generate G(z, c),', 'keep the one closest to t'],
     ['BERT training', 'clean training sentences only'],
     ['InfoGAN training (generator G, helper Q)', 'on embeddings of clean training sentences.', 'Never sees an attack.'],
     ('Random', 'a new projection on every query'), False)
lane(334, 'Ours (notebook 07b)',
     ['BERT', 'adversarially', 'trained'],
     ['Learned encoder', 'E(t, c) predicts z in one pass,', 'then G(z, c)'],
     ['BERT training: clean + attacked training', 'sentences (black-box attacks, 2 rounds)', '+ random-noise copies; taught to give the',
      'same answer and embedding for both'],
     ['InfoGAN: the paper\'s Table 6 settings, new embeddings.', 'Encoder (Cengiz\'s design), G frozen: send attacked',
      'embeddings back to their clean point; plus', 'consistency, reconstruction and label losses.'],
     ('Deterministic', 'same input, same output'), True)

box(20, 612, 1160, 34, ['Evaluation, identical for both: the authors\' attack code (PWWS, TextFooler, BERT-Attack), '
                        '1000 test sentences, at most 30% of words changed, sentence similarity ≥ 0.84'],
    SAME, size=13, bold_first=False)

text(20, 690, 'What is the embedding t?', 18, 'bold')
for i, s in enumerate(['BERT turns each sentence into 768 numbers', 'that summarise its meaning: a point in space.',
                       'The classifier only ever sees this point.', '',
                       'Clean sentences of one topic lie on a curved', 'band (the "manifold"); the GAN learns the bands.',
                       '', 'An attack swaps a few words, so the point moves', 'off its band and the classifier is fooled.',
                       '', 'Schematic, not real data.']):
    text(20, 716 + i * 18, s, 13, 'normal', 'start', MUTED if i == 10 else INK, 'italic' if i == 10 else 'normal')
panel(470, 720, '1. An attack moves t', 'attack')
panel(710, 720, '2. TMD: random guesses', 'tmd')
panel(950, 720, '3. Ours', 'ours')
for cx, (l1, l2) in zip((580, 820, 1060), [('pushed off its band,', 'near the wrong topic'),
                                           ('closest of 30 guesses on a band;', 'a new pick on every query'),
                                           ('robust BERT moves it less;', 'the encoder maps it straight back')]):
    text(cx, 890, l1, 12, 'normal', 'middle', MUTED)
    text(cx, 906, l2, 12, 'normal', 'middle', MUTED)
out.append('</svg>')
open(sys.argv[1], 'w').write('\n'.join(out))
