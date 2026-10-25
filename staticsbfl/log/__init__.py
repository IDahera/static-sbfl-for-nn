"""Everything that writes to the console or a log file, in one place.

``console``     the format, banners, step headers, durations, sizes, tables and
                file logs every script uses
``pipeline``    what a phase does and writes, its status and its runtime
                estimate
``evaluation``  the console/log rendering of the final evaluation: accuracy
                summaries, the cost report and the runtime comparison

Modules here format and print; they compute nothing that is not also
available from :mod:`staticsbfl.evaluation` as plain data.
"""

from staticsbfl.log.console import (
    RULE_WIDTH, Step, add_file_log, configure_logging, display_path, format_duration,
    human_bytes, human_count, labelled, log_table, phase_banner, phase_done, plain, style,
    wrap_text,
)

__all__ = [
    "RULE_WIDTH", "Step", "add_file_log", "configure_logging", "display_path",
    "format_duration", "human_bytes", "human_count", "labelled", "log_table",
    "phase_banner", "phase_done", "plain", "style", "wrap_text",
]
