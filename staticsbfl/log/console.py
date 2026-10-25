"""Console logging shared by the scripts (``staticsbfl.log.console``).

One format, set up once per script entry point, so the progress output of the
training and suspiciousness runs reads the same way.

Two kinds of line go through the same handler. Ordinary log records carry a
timestamp and a level. *Structure* -- the banner that opens a phase, the header
of one step inside it, the line that closes it -- is logged as a ``plain``
record and printed without that prefix, so a box drawn around a phase stays a
box. Both go through ``logging`` rather than ``print`` so they can never come
out of order with each other.
"""

import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

#: Width of every banner and rule, in characters.
RULE_WIDTH = 78

_RESET = "\033[0m"
_STYLES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
}
_LEVEL_STYLE = {"WARNING": "yellow", "ERROR": "red", "CRITICAL": "red"}


def _colour_enabled(stream) -> bool:
    """ANSI colour only on a terminal, and never when ``NO_COLOR`` is set."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


_COLOUR = _colour_enabled(sys.stdout)


def style(text: str, *names: str) -> str:
    """Wrap *text* in the named ANSI styles, or return it unchanged."""
    if not _COLOUR or not names:
        return text
    return "".join(_STYLES[n] for n in names) + text + _RESET


class _ConsoleFormatter(logging.Formatter):
    """Timestamped records, except ``plain`` ones, which print as they are."""

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "plain", False):
            return record.getMessage()
        line = super().format(record)
        level_style = _LEVEL_STYLE.get(record.levelname)
        return style(line, level_style) if level_style else line


def configure_logging(verbose: bool = False) -> None:
    """Install a single stream handler on the root logger."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _ConsoleFormatter("%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    )
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=[handler],
        force=True,
    )
    # The dataset download path is chatty -- one INFO line per HTTP request,
    # plus cache-hit chatter -- and it drowns out the training progress.
    for noisy in ("datasets", "filelock", "httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # torchvision's CIFAR-10 unpickler trips a NumPy >= 2.4 deprecation on every
    # load. It is theirs to fix, says nothing about this run, and lands in the
    # middle of the step output.
    warnings.filterwarnings("ignore", message=r"dtype\(\): align should be passed")


# --------------------------------------------------------------------------
# Structure: phases, steps, outputs
# --------------------------------------------------------------------------

#: One output to point at: a path, or ``(path, what it holds)``.
Output = Union[str, Path, Tuple[Union[str, Path], str]]


def display_path(path: Union[str, Path]) -> str:
    """*path* relative to the working directory where possible -- what you would type."""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def format_duration(seconds: float) -> str:
    """``'42s'``, ``'7m 05s'``, ``'2h 13m'`` -- the precision a reader needs."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s" if seconds >= 10 else f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def plain(logger: logging.Logger, text: str = "") -> None:
    """Log *text* line by line without the timestamp prefix."""
    for line in text.split("\n"):
        logger.info(line, extra={"plain": True})


def _output_lines(outputs: Iterable[Output], indent: str) -> list:
    """``▸ path   what it holds``, paths aligned, in the highlight colour."""
    items = []
    for item in outputs:
        path, note = (item if isinstance(item, tuple) else (item, ""))
        items.append((display_path(path), note))
    if not items:
        return []
    width = max(len(p) for p, _ in items)
    return [
        f"{indent}{style('▸ ' + p.ljust(width), 'yellow')}"
        + (f"   {style(note, 'dim')}" if note else "")
        for p, note in items
    ]


def labelled(label: str, lines: Sequence[str], indent: str = "  ", width: int = 10) -> list:
    """A two-column block: *label* beside the first line, the rest aligned under it."""
    out = []
    for index, line in enumerate(lines):
        head = style(label.ljust(width), "bold") if index == 0 else " " * width
        out.append(f"{indent}{head} {line}")
    return out


def phase_banner(
    logger: logging.Logger,
    title: str,
    rows: Sequence[Tuple[str, Sequence[str]]] = (),
    outputs: Iterable[Output] = (),
) -> None:
    """The double-ruled box that opens a phase, plus its labelled facts.

    *rows* is ``[(label, lines), ...]``, printed under the box in order;
    *outputs* is listed last under ``Results``, highlighted, because "where do I
    look afterwards" is the question a long run leaves you with.
    """
    inner = RULE_WIDTH - 4
    lines = [
        "",
        style("╔" + "═" * (RULE_WIDTH - 2) + "╗", "cyan"),
        style("║ ", "cyan") + style(title[:inner].ljust(inner), "bold") + style(" ║", "cyan"),
        style("╚" + "═" * (RULE_WIDTH - 2) + "╝", "cyan"),
    ]
    for label, block in rows:
        lines += labelled(label, list(block))
    output_lines = _output_lines(outputs, "")
    if output_lines:
        lines += labelled("Results", output_lines)
    lines.append("")
    plain(logger, "\n".join(lines))


def phase_done(
    logger: logging.Logger,
    title: str,
    elapsed: Optional[float] = None,
    rows: Sequence[Tuple[str, Sequence[str]]] = (),
    outputs: Iterable[Output] = (),
    ok: bool = True,
) -> None:
    """The line that closes a phase, and where its results now are."""
    mark = style("✔", "green", "bold") if ok else style("✘", "red", "bold")
    took = f" in {format_duration(elapsed)}" if elapsed is not None else ""
    visible = f"══ ✔ {title}{took} "
    fill = "═" * max(3, RULE_WIDTH - len(visible))
    lines = ["", style("══ ", "cyan") + mark + style(f" {title}{took} ", "bold") + style(fill, "cyan")]
    for label, block in rows:
        lines += labelled(label, list(block))
    output_lines = _output_lines(outputs, "")
    if output_lines:
        lines += labelled("Results", output_lines)
    lines.append(style("═" * RULE_WIDTH, "cyan"))
    lines.append("")
    plain(logger, "\n".join(lines))


class Step:
    """One step of a phase: a header saying what it does, a footer saying it is done.

        with Step(logger, 2, 4, "score", "...", outputs=[...], context="phase1") as step:
            ...
            step.summary = "120 scored, 0 skipped"

    The footer reports the elapsed time and repeats the outputs, so the end of
    a long scrollback says where to look without scrolling back to its start.
    An exception closes the step as failed and propagates unchanged.
    """

    def __init__(
        self,
        logger: logging.Logger,
        index: int,
        total: int,
        name: str,
        what: Union[str, Sequence[str]],
        outputs: Iterable[Output] = (),
        context: str = "",
    ):
        self.logger = logger
        self.label = f"Step {index}/{total} · {name}"
        self.what = [what] if isinstance(what, str) else list(what)
        self.outputs = list(outputs)
        self.context = context
        self.summary = ""
        self.started = 0.0

    def __enter__(self) -> "Step":
        self.started = time.perf_counter()
        head = f"┌─ {self.label} " + (f"─ {self.context} " if self.context else "")
        lines = ["", style(head, "cyan", "bold") + style("─" * max(3, RULE_WIDTH - len(head)), "cyan")]
        bar = style("│", "cyan")
        lines += [f"{bar}  {line}" for line in self.what]
        for index, line in enumerate(_output_lines(self.outputs, "")):
            lines.append(f"{bar}  {style('Writes', 'bold') if index == 0 else '      '}  {line}")
        lines.append(style("└" + "─" * (RULE_WIDTH - 1), "cyan"))
        plain(self.logger, "\n".join(lines))
        return self

    def section(self, title: str) -> None:
        """A sub-heading inside the step, for its own sub-parts."""
        plain(self.logger, "")
        plain(self.logger, style(f"  ── {title} ", "bold") + style("─" * max(3, RULE_WIDTH - len(title) - 6), "dim"))

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def __exit__(self, exc_type, exc, _tb) -> bool:
        ok = exc_type is None
        mark = style("✔", "green", "bold") if ok else style("✘", "red", "bold")
        status = "done" if ok else "FAILED"
        line = f"{mark} {self.label} {status} in {format_duration(self.elapsed)}"
        if self.summary and ok:
            line += f" — {self.summary}"
        lines = ["", line]
        if ok:
            lines += _output_lines(self.outputs, "   ")
        plain(self.logger, "\n".join(lines))
        return False


def human_bytes(n: int) -> str:
    """Format a byte count as a short string, e.g. ``'4.3 MB'``."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def human_count(n: int) -> str:
    """Format a parameter count compactly, e.g. ``'11.2 M'``."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} k"
    return str(n)


# --------------------------------------------------------------------------
# Generic helpers: file logs, wrapped text, tables
# --------------------------------------------------------------------------


def add_file_log(path: Union[str, Path], *, mode: str = "a", timestamps: bool = True) -> logging.Handler:
    """Also write every record to *path*, uncoloured; returns the handler.

    The console handler keeps its colours; the file gets the same lines without
    escape codes, so a log can be read with ``less`` and quoted in a write-up.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
    handler.setFormatter(_PlainFormatter(timestamps))
    logging.getLogger().addHandler(handler)
    return handler


class _PlainFormatter(logging.Formatter):
    """The console format without colour codes, for log files."""

    def __init__(self, timestamps: bool) -> None:
        super().__init__("%(asctime)s  %(levelname)-7s %(message)s" if timestamps else "%(message)s",
                         datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "plain", False):
            text = record.getMessage()
        else:
            text = super().format(record)
        return _strip_styles(text)


def _strip_styles(text: str) -> str:
    for code in list(_STYLES.values()) + [_RESET]:
        text = text.replace(code, "")
    return text


def wrap_text(text: str, width: int = 76) -> list:
    """Greedy word wrap into lines of at most *width* characters."""
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


def log_table(
    logger: logging.Logger,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    align: Optional[str] = None,
    indent: str = "    ",
    highlight: Optional[Sequence[bool]] = None,
) -> None:
    """Print *rows* under *headers* as an aligned plain-text table.

    *align* is one character per column, ``l`` or ``r`` (default: first column
    left, the rest right). Cells are printed with ``str``; ``None`` prints as
    ``-``. Rows whose *highlight* entry is true are printed in yellow.
    """
    cells = [["-" if c is None else str(c) for c in row] for row in rows]
    widths = [max([len(str(h))] + [len(r[i]) for r in cells]) for i, h in enumerate(headers)]
    align = align or ("l" + "r" * (len(headers) - 1))

    def line(values):
        return indent + "  ".join(
            v.ljust(w) if a == "l" else v.rjust(w) for v, w, a in zip(values, widths, align)
        )

    plain(logger, style(line([str(h) for h in headers]), "dim"))
    for index, row in enumerate(cells):
        text = line(row)
        plain(logger, style(text, "yellow") if highlight and highlight[index] else text)
