"""Figure typography for every paper figure.

v9: Gentium Book Basic only. v13: Latin letters, digits and math stay in
Gentium Book Basic; Hangul (which Gentium lacks) falls back to NanumMyeongjo,
the same family as the paper body text. Every embedded PDF font must be one of
these two families.

- Registers the bundled SIL OFL fonts in fonts/gentium-book-basic/ (no system
  install needed; a system copy is used only if the bundle is missing).
- Sets text and mathtext (custom fontset) to Gentium Book Basic and disables
  mathtext fallback, so a glyph the font lacks raises instead of silently
  switching to DejaVu/Computer Modern. Gentium Book Basic has no Greek letters
  or the \\preceq symbol, so figure labels spell those out (e.g. "lambda").
- PDFs embed TrueType (Type 42) subsets; check_pdf_fonts() verifies that every
  embedded font in a PDF is Gentium Book Basic.
"""
from pathlib import Path
import logging, re, warnings
import matplotlib
from matplotlib import font_manager

FAMILY = 'Gentium Book Basic'
KO_FAMILY = 'NanumMyeongjo'
ALLOWED_PDF_FONTS = ('GentiumBookBasic', 'NanumMyeongjo')
ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'fonts' / 'gentium-book-basic'
FILES = {'regular': 'GenBkBasR.ttf', 'italic': 'GenBkBasI.ttf', 'bold': 'GenBkBasB.ttf', 'bolditalic': 'GenBkBasBI.ttf'}
KO_BUNDLE = ROOT / 'fonts' / 'nanum-myeongjo'
KO_FILES = {'regular': 'NanumMyeongjo.ttf', 'bold': 'NanumMyeongjoBold.ttf'}
KO_SYSTEM_DIRS = [Path('/usr/share/fonts/truetype/nanum'), Path.home() / 'Library/Fonts', Path('/Library/Fonts')]
SYSTEM_DIRS = [Path('/usr/share/fonts/truetype/gentium-basic'), Path.home() / 'Library/Fonts', Path('/Library/Fonts')]


def _register():
    found = {}
    for style, name in FILES.items():
        for d in [BUNDLE, *SYSTEM_DIRS]:
            p = d / name
            if p.exists():
                font_manager.fontManager.addfont(str(p))
                found[style] = p
                break
    missing = sorted(set(FILES) - set(found))
    if missing:
        raise FileNotFoundError(f'{FAMILY} font files not found for {missing}; expected in {BUNDLE}')
    for style, name in KO_FILES.items():
        for d in [KO_BUNDLE, *KO_SYSTEM_DIRS]:
            p = d / name
            if p.exists():
                font_manager.fontManager.addfont(str(p))
                found['ko_' + style] = p
                break
        else:
            raise FileNotFoundError(f'{KO_FAMILY} {name} not found; expected in {KO_BUNDLE}')
    return found


def apply(base_size=9):
    """Register fonts and set rcParams. Returns {style: font path}."""
    found = _register()
    for style, prop in [('regular', dict(style='normal', weight='normal')), ('italic', dict(style='italic', weight='normal')),
                        ('bold', dict(style='normal', weight='bold')), ('bolditalic', dict(style='italic', weight='bold'))]:
        resolved = font_manager.findfont(font_manager.FontProperties(family=FAMILY, **prop), fallback_to_default=False)
        if Path(resolved).name != FILES[style]:
            raise RuntimeError(f'{FAMILY} {style} resolved to {resolved}')
    ko = font_manager.findfont(font_manager.FontProperties(family=KO_FAMILY), fallback_to_default=False)
    if Path(ko).name != KO_FILES['regular']:
        raise RuntimeError(f'{KO_FAMILY} resolved to {ko}')
    matplotlib.rcParams.update({
        'font.family': [FAMILY, KO_FAMILY], 'font.serif': [FAMILY, KO_FAMILY], 'font.size': base_size,
        'mathtext.fontset': 'custom', 'mathtext.rm': FAMILY, 'mathtext.it': f'{FAMILY}:italic',
        'mathtext.bf': f'{FAMILY}:bold', 'mathtext.bfit': f'{FAMILY}:bold:italic', 'mathtext.sf': FAMILY,
        'mathtext.tt': FAMILY, 'mathtext.cal': f'{FAMILY}:italic', 'mathtext.fallback': None,
        'mathtext.default': 'it', 'axes.unicode_minus': True,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })
    return found


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def savefig(fig, path, **kw):
    """fig.savefig that fails if any glyph is missing from Gentium Book Basic.

    Matplotlib normally substitutes a dummy glyph with only a warning/log entry;
    here that becomes an error so no figure silently loses a character.
    """
    handler = _Collect()
    logger = logging.getLogger('matplotlib')
    logger.addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fig.savefig(path, **kw)
    finally:
        logger.removeHandler(handler)
    problems = [m for m in handler.messages + [str(w.message) for w in caught]
                if 'glyph' in m.lower() or 'findfont' in m.lower()]
    if problems:
        raise RuntimeError(f'{path}: font problem(s): {problems}')


def pdf_fonts(path):
    """Base font names embedded in a matplotlib PDF (subset prefixes removed)."""
    data = Path(path).read_bytes()
    names = {m.decode('latin-1') for m in re.findall(rb'/BaseFont\s*/([^\s/<>\[\]]+)', data)}
    return sorted({n.split('+', 1)[-1] for n in names})


def check_pdf_fonts(path):
    names = pdf_fonts(path)
    if not names:
        raise AssertionError(f'{path}: no embedded font found')
    bad = [n for n in names if not n.replace(' ', '').startswith(ALLOWED_PDF_FONTS)]
    if bad:
        raise AssertionError(f'{path}: fonts other than Gentium Book Basic / NanumMyeongjo: {bad}')
    return names
