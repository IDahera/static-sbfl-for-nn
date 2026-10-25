"""How much of the disk ``cache/`` occupies, broken down by what produced it.

    make cache-size
    python -m staticsbfl.cli.cache_size

Everything this project generates lives under ``cache/``, so this is the whole
footprint. It reads nothing but file sizes and writes nothing, so it is safe to
run at any time -- including next to a running experiment.

The areas are listed in the order of the pipeline, each with the ``make`` target
that would delete it, because "how big is it?" is usually asked right before
"can I delete it?".
"""

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import click

from staticsbfl.log.console import configure_logging, display_path, plain, style
from staticsbfl.utils.paths import (
    CACHE_DIR, DATA_DIR, DEBUG_VARIANTS_DIR, HEAVY_VARIANTS_DIR, RUNTIME_DIR, VARIANTS_DIR,
)

logger = logging.getLogger("cache-size")


def tree_size(path: Path, include: Optional[Callable[[Path], bool]] = None) -> Tuple[int, int]:
    """``(bytes, files)`` under *path*; ``(0, 0)`` if it does not exist.

    *include*, if given, decides per top-level entry whether it counts -- which
    is how the legacy phase directories are told apart from the per-dataset
    trees that share their parent.
    """
    path = Path(path)
    if not path.exists():
        return 0, 0
    if path.is_file():
        return path.stat().st_size, 1
    total, files = 0, 0
    for entry in path.iterdir():
        if include is not None and not include(entry):
            continue
        if entry.is_file():
            total, files = total + entry.stat().st_size, files + 1
            continue
        for root, _dirs, names in os.walk(entry):
            for name in names:
                try:
                    total += os.stat(os.path.join(root, name)).st_size
                    files += 1
                except OSError:  # a file removed while walking
                    pass
    return total, files


def human(n: int) -> str:
    """``'4.3 GB'`` -- decimal units, as Finder and ``df -H`` show them."""
    size = float(n)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} TB"


@dataclass(frozen=True)
class Area:
    label: str
    path: Path
    clean: str                     # the make target that deletes it
    include: Optional[Callable[[Path], bool]] = None


def _datasets(root: Path) -> List[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []


def areas() -> List[Tuple[str, List[Area]]]:
    """Every part of the cache, grouped by who produced it."""
    legacy = [p for p in sorted(VARIANTS_DIR.glob("phase*")) if p.is_dir()] if VARIANTS_DIR.exists() else []
    populations: List[Area] = []
    for mode, root in (("light", VARIANTS_DIR), ("heavy", HEAVY_VARIANTS_DIR), ("debug", DEBUG_VARIANTS_DIR)):
        for tree in _datasets(root):
            if tree.name.startswith("phase"):
                continue  # the legacy layout, listed on its own below
            target = "clean-debug" if mode == "debug" else f"clean-variants VMODE={mode} DATASET={tree.name} PHASE=..."
            populations.append(Area(f"{mode:<6}{tree.name}", tree, target))
    if legacy:
        populations.append(Area("legacy layout (read by nothing)", VARIANTS_DIR,
                       "rm -rf cache/variants/phase?", include=lambda p: p in legacy))

    return [
        ("Downloads", [
            Area("datasets", DATA_DIR, "clean   (no scoped target)"),
        ]),
        ("Architecture-variation populations", populations),
        ("Runtime comparison", [
            Area("exports, conversions, records", RUNTIME_DIR, "clean-runtime"),
        ]),
    ]


def print_cache_sizes(log: logging.Logger = logger) -> None:
    plain(log, "")
    plain(log, style(f"Cache footprint · {display_path(CACHE_DIR)}/", "bold", "cyan"))
    if not CACHE_DIR.exists():
        plain(log, "  cache/ does not exist -- nothing has been computed yet.")
        _print_disk(log)
        return

    grand = 0
    for group, members in areas():
        plain(log, "")
        plain(log, style(f"  {group}", "bold"))
        if not members:
            plain(log, style("    (nothing yet)", "dim"))
        for area in members:
            size, files = tree_size(area.path, area.include)
            grand += size
            shown = style(f"{human(size):>9}", "yellow") if size else style(f"{'–':>9}", "dim")
            plain(log, f"    {area.label:<34}{shown}  {files:>7,} files   "
                       + style(f"make {area.clean}" if not area.clean.startswith("rm") else area.clean, "dim"))

    total, files = tree_size(CACHE_DIR)
    other = total - grand
    plain(log, "")
    if other > 1_000_000:
        plain(log, f"  {'other':<36}{human(other):>9}")
    plain(log, style(f"  {'total cache/':<36}{human(total):>9}  {files:>7,} files   make clean", "bold"))
    _print_disk(log)


def _print_disk(log: logging.Logger) -> None:
    usage = shutil.disk_usage(CACHE_DIR if CACHE_DIR.exists() else CACHE_DIR.parent)
    plain(log, style(f"  {'free on this disk':<36}{human(usage.free):>9}  of {human(usage.total)}", "dim"))
    plain(log, "")


@click.command()
def main():
    """Print the size of every part of cache/, and what deletes it."""
    configure_logging()
    print_cache_sizes()


if __name__ == "__main__":
    main()
