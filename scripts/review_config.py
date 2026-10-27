"""Shared paths and portable CJK font resolution for the candidate workflow."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
def output_argument(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if '--output' in argv:
        i=argv.index('--output')
        if i+1>=len(argv):raise ValueError('--output requires a directory')
        return argv[i+1]
    for arg in argv:
        if arg.startswith('--output='):return arg.split('=',1)[1]
    return os.environ.get('MATHORCUP_OUTPUT',str(ROOT/'outputs/competition_candidate_v3'))

OUT=Path(output_argument()).resolve()
FROZEN = ROOT/'frozen_inputs'
VERSION = 'v3'


def chinese_font(explicit=None):
    """Choose a TrueType CJK face supported by both ReportLab and Matplotlib.

    A caller-supplied path is authoritative: invalid paths never silently fall back.
    Verify actual glyph coverage rather than trusting a font filename.
    """
    from reportlab.pdfbase.ttfonts import TTFont
    selected = explicit or os.environ.get('MATHORCUP_CJK_FONT')
    candidates = [Path(selected)] if selected else [
        ROOT/'fonts/NotoSansSC-Regular.ttf',
        ROOT/'fonts/NotoSansCJKsc-Regular.ttf',
        Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
        Path('/usr/share/fonts/truetype/arphic/uming.ttc'),
        Path('/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf'),
        Path('/Library/Fonts/Arial Unicode.ttf'),
        Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts/simhei.ttf',
        Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts/msyh.ttc',
    ]
    problems=[]
    for p in candidates:
        if not p.is_file():
            problems.append(str(p)+': missing')
            continue
        try:
            font=TTFont('CJKProbe',str(p),subfontIndex=0)
            missing=[ch for ch in '啤酒评分模型约束审查' if ord(ch) not in font.face.charToGlyph]
            if missing: raise ValueError('missing CJK glyphs: '+''.join(missing))
            return p.resolve()
        except Exception as exc:
            problems.append(str(p)+': '+str(exc))
    raise FileNotFoundError(
        '找不到可用于PDF及图表的中文TrueType字体。请用 --font /path/to/CJK.ttf '
        '或 MATHORCUP_CJK_FONT 指定合法字体；也可放入 fonts/NotoSansSC-Regular.ttf。'
        '\n不支持仅有西文字形的字体；部分CFF格式OTF不受ReportLab支持。\n'+ '\n'.join(problems))
