"""Runtime and memory of static prediction vs. dynamic hit spectra.

    make runtime                               # light, CIFAR-10, phase-4 designed models
    python -m staticsbfl.cli.runtime_benchmark --mode heavy
    python -m staticsbfl.cli.runtime_benchmark --tool hs_pt_cpu --tool deepfault_tf2 --max-samples 500

Applies every tool to the 20 *designed* (established) models of phase 4 and
measures each one in its own fresh process -- one process per (tool, model,
repeat), so no two measurements share an interpreter, a framework or a cache:

    staticsbfl_<predictor>  the phase-4 predictor (fitted on the sampled
                            population) applied to the model's weights; no
                            data, no forward pass               -- runtime_static.py
    hs_pt_cpu / _mps      this repository's streaming PyTorch spectrum, as make score
                            runs it (one pass, counts per batch) -- runtime_hs.py
    icsoft_pt_cpu / _mps    opt-in: the published ICSOFT 2025 code (stores every
                            activation, one pass per layer)     -- runtime_icsoft.py
    deepfault_tf1 / _tf2    the two DeepFault ports, upstream functions, in
                            their own conda environments        -- baselines/deepfault_*/

Steps (each skips work already on disk unless ``--force``, so an interrupted
run resumes where it stopped):

    1 export    the pre-processed test set and every model's weights as plain
                ``.npz`` (framework-neutral), plus the PyTorch predictions
    2 keras     each DeepFault port converts the weights into its own Keras
                format and checks it reproduces the PyTorch predictions
    3 measure   the runs; failures and timeouts are recorded, never fatal
    4 fidelity  the counts each dynamic tool produced vs. the spectrum the score step
                cached and vs. HS@pt on the CPU
    5 table     ``runs.csv`` and ``fidelity.csv`` for the evaluation

Nothing here summarises: ``make evaluate`` reads the CSVs at the end.
"""

import csv
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import click
import numpy as np

from staticsbfl.benchmark.footprint import EXACT_MALLOC_ENV, MEASUREMENT_NOTE, SOURCE as MEMORY_SOURCE
from staticsbfl.evaluation.runtime import (
    DYNAMIC_TOOLS, EVAL_SPLIT, FIDELITY_COLUMNS, METRICS, PHASE, TOOL_LABELS,
    compare_spectra, flatten, run_columns,
)
from staticsbfl.log.console import (
    Step, add_file_log, configure_logging, display_path, format_duration, human_bytes, phase_banner, plain,
    style, wrap_text,
)
from staticsbfl.utils.paths import (
    PROJECT_ROOT, ensure_dir, runtime_root, runtime_test_set, variant_model_path,
    variant_predictions_dir, variant_spectrum_path,
)
from staticsbfl.variation import build_phase
from staticsbfl.variation.phases import POPULATION_SIZES

logger = logging.getLogger("runtime")

DATASETS = ("cifar10", "fashion_mnist")
#: What a plain run measures. ICSoft@pt (the published ICSOFT code) is opt-in,
#: ``--tool icsoft_pt_cpu``: HS@pt is the same idea in its current form, and
#: the comparison the paper needs is static prediction vs. computing the spectrum.
DEFAULT_TOOLS = (
    "staticsbfl_gradient_boosting", "staticsbfl_mlp",
    "hs_pt_cpu", "hs_pt_mps",
    "deepfault_tf1", "deepfault_tf2",
)
ALL_TOOLS = tuple(TOOL_LABELS)
PORTS = ("tf1", "tf2")

#: Name of the conda environment of each DeepFault port (``make runtime-setup``).
CONDA_ENVS = {port: f"deepfault-{port}" for port in PORTS}


def conda_envs_dir() -> Optional[Path]:
    """The ``envs`` directory of the conda installation this interpreter belongs to."""
    prefix = Path(sys.prefix)
    if prefix.parent.name == "envs":
        return prefix.parent
    if (prefix / "envs").is_dir():
        return prefix / "envs"
    try:
        info = json.loads(subprocess.run(["conda", "info", "--json"], capture_output=True, text=True).stdout)
        return Path(info["envs_dirs"][0])
    except (OSError, ValueError, KeyError, IndexError):
        return None


def port_python(port: str) -> Path:
    """Interpreter of a DeepFault port: conda env ``deepfault-<port>``, or ``$DEEPFAULT_<PORT>_PYTHON``."""
    override = os.environ.get(f"DEEPFAULT_{port.upper()}_PYTHON")
    if override:
        return Path(override)
    envs = conda_envs_dir()
    return (envs / CONDA_ENVS[port] / "bin" / "python") if envs else Path(f"<no conda>/{CONDA_ENVS[port]}")


# --------------------------------------------------------------------------
# 1 export
# --------------------------------------------------------------------------


def export_test_set(dataset: str, force: bool) -> Path:
    """The test split exactly as make score feeds it to the models, as one array."""
    from staticsbfl.data import load_dataset

    path = runtime_test_set(dataset)
    if path.exists() and not force:
        logger.info("test set cached: %s", display_path(path))
        return path
    bundle = load_dataset(dataset, batch_size=1000)
    xs, ys = [], []
    for inputs, targets in bundle.test_loader:
        xs.append(inputs.numpy().astype(np.float32))
        ys.append(targets.numpy().astype(np.int64))
    ensure_dir(path.parent)
    np.savez(path, x=np.concatenate(xs), y=np.concatenate(ys), num_classes=bundle.num_classes)
    logger.info("wrote %s (%d samples, %s)", display_path(path), sum(len(y) for y in ys),
                human_bytes(path.stat().st_size))
    return path


#: Seeds per designed shape the runtime comparison measures by default. Runtime
#: depends on the architecture, not on the seed, so more seeds only repeat the
#: same measurement -- and DeepFault costs hours per seed set.
DEFAULT_SEEDS_PER_DESIGN = 4


def seed_index(key: str) -> int:
    """``p4_designed_wide_mlp_s7`` -> 7."""
    return int(key.rsplit("_s", 1)[1])


def export_models(mode: str, dataset: str, test_set: Path, keys: Sequence[str], force: bool,
                  seeds_per_design: int = DEFAULT_SEEDS_PER_DESIGN) -> List[dict]:
    """Weights as ``.npz`` + a JSON description per model; the PyTorch argmax as reference.

    Only the first *seeds_per_design* seeds of every designed shape (0 = all).
    """
    import torch

    from staticsbfl.persistence import load_model
    from staticsbfl.variation import build_model

    spec = build_phase(PHASE, mode)
    out = ensure_dir(runtime_root(mode, dataset) / "weights")
    x = torch.from_numpy(np.load(test_set)["x"])
    exported = []
    for config in spec.configs():
        if config.split != EVAL_SPLIT or (keys and config.key not in keys):
            continue
        if seeds_per_design and not keys and seed_index(config.key) >= seeds_per_design:
            continue
        checkpoint_path = variant_model_path(PHASE, config.split, config.key, mode, dataset)
        if not checkpoint_path.exists():
            logger.warning("no checkpoint for %s; train phase 4 first (make phase PHASE=phase4)", config.key)
            continue
        meta = {
            "config": {
                "key": config.key, "family": config.family, "hidden_widths": list(config.hidden_widths),
                "activation": config.activation, "seed": config.seed, "phase": config.phase,
                "split": config.split,
            },
            "hidden_widths": list(config.hidden_widths), "activation": config.activation,
            "checkpoint": str(checkpoint_path),
            "spectrum": str(variant_spectrum_path(PHASE, config.split, config.key, mode, dataset)),
        }
        npz_path = out / f"{config.key}.npz"
        fresh = npz_path.exists() and npz_path.stat().st_mtime >= checkpoint_path.stat().st_mtime
        checkpoint = load_model(checkpoint_path)
        if not fresh or force:
            model = build_model(config, num_classes=checkpoint.num_classes, input_shape=checkpoint.input_shape)
            model.load_state_dict(checkpoint.state_dict)
            model.eval()
            with torch.no_grad():
                reference = torch.cat([model(chunk).argmax(dim=1) for chunk in x.split(1000)]).numpy()
            np.savez(npz_path, reference_argmax=reference,
                     **{k: v.numpy() for k, v in checkpoint.state_dict.items()})
            logger.info("exported %s (%s)", config.key, "-".join(map(str, config.hidden_widths)))
        meta["input_shape"] = list(checkpoint.input_shape)
        meta["num_classes"] = int(checkpoint.num_classes)
        (out / f"{config.key}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        exported.append(meta)
    return exported


def exported_metas(mode: str, dataset: str) -> List[dict]:
    """Every model exported so far, whichever invocation exported it."""
    metas = []
    for path in sorted((runtime_root(mode, dataset) / "weights").glob("*.json")):
        try:
            metas.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            logger.warning("unreadable export description %s", display_path(path))
    return metas


# --------------------------------------------------------------------------
# 2 keras
# --------------------------------------------------------------------------


def convert_for_ports(mode: str, dataset: str, test_set: Path, ports: Sequence[str], force: bool) -> Dict[str, bool]:
    """Run each port's ``torch_bridge.py`` in its own environment."""
    ok = {}
    root = runtime_root(mode, dataset)
    for port in ports:
        python = port_python(port)
        if not python.exists():
            logger.error("DF@%s: no interpreter at %s -- run `make runtime-setup`.", port, python)
            ok[port] = False
            continue
        command = [
            str(python), str(PROJECT_ROOT / "baselines" / f"deepfault_{port}" / "torch_bridge.py"),
            "--weights-dir", str(root / "weights"), "--out-dir", str(root / f"keras_{port}"),
            "--test-set", str(test_set),
        ] + (["--force"] if force else [])
        log_path = root / f"keras_{port}.log"
        with open(log_path, "w", encoding="utf-8") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                    env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"})
        for line in log_path.read_text().splitlines():
            if line.startswith(("exported", "skip")):
                logger.info("DF@%s  %s", port, line)
        ok[port] = result.returncode == 0
        if not ok[port]:
            logger.error("DF@%s conversion failed or disagreed with PyTorch; see %s", port, display_path(log_path))
    return ok


# --------------------------------------------------------------------------
# 3 measure
# --------------------------------------------------------------------------


def run_tag(args: dict) -> str:
    """Name of a sample scope: 'all', 'class3', 'all_first500', ...

    Records of different scopes live apart, so a smoke run on 500 samples can
    never be mistaken for -- or skipped as -- a measurement on the full set.
    """
    tag = "all" if args["samples"] == "all" else f"class{args['samples']}"
    return tag + (f"_first{args['max_samples']}" if args["max_samples"] else "")


def runs_dir(args: dict, tool: str) -> Path:
    return runtime_root(args["mode"], args["dataset"]) / "runs" / run_tag(args) / tool


def table_path(args: dict, name: str) -> Path:
    """``runs.csv`` for the full test set, ``runs__<tag>.csv`` otherwise."""
    tag = run_tag(args)
    return runtime_root(args["mode"], args["dataset"]) / (f"{name}.csv" if tag == "all" else f"{name}__{tag}.csv")


def estimator_dir(mode: str, dataset: str) -> Path:
    return variant_predictions_dir(PHASE, mode, dataset) / "estimators"


def command_for(tool: str, meta: dict, args: dict, record: Path, artefact: Path) -> List[str]:
    """The command line of one measured process."""
    key = meta["config"]["key"]
    root = runtime_root(args["mode"], args["dataset"])
    weights = str(root / "weights" / f"{key}.json")
    test_set = str(runtime_test_set(args["dataset"]))
    samples = ["--selected-class", args["samples"]] + (
        ["--max-samples", str(args["max_samples"])] if args["max_samples"] else [])
    if tool.startswith("staticsbfl_"):
        return [sys.executable, "-m", "staticsbfl.cli.runtime_static", "--meta", weights,
                "--predictor", tool[len("staticsbfl_"):], "--estimator-dir", str(estimator_dir(args["mode"], args["dataset"])),
                "--target-kind", args["target_kind"], "--record", str(record), "--predictions-out", str(artefact)]
    for prefix, module in (("hs_pt_", "staticsbfl.cli.runtime_hs"), ("icsoft_pt_", "staticsbfl.cli.runtime_icsoft")):
        if tool.startswith(prefix):
            return [sys.executable, "-m", module, "--meta", weights, "--test-set", test_set,
                    "--device", tool[len(prefix):], "--record", str(record), "--spectrum-out", str(artefact)] + samples
    port = tool[len("deepfault_"):]
    return [str(port_python(port)), str(PROJECT_ROOT / "baselines" / f"deepfault_{port}" / "runtime_run.py"),
            "--model", str(root / f"keras_{port}" / key), "--test-set", test_set,
            "--record", str(record), "--spectrum-out", str(artefact)] + samples


def tool_available(tool: str, args: dict, keras_ok: Dict[str, bool]) -> Optional[str]:
    """``None`` when *tool* can run, else the reason it cannot."""
    if tool.startswith("staticsbfl_"):
        predictor = tool[len("staticsbfl_"):]
        directory = estimator_dir(args["mode"], args["dataset"])
        target = lambda m: m if args["target_kind"] == "raw" else f"{m}_{args['target_kind']}"
        missing = [m for m in METRICS if not (directory / f"{PHASE}__{m}__{target(m)}__{predictor}.joblib").exists()]
        if missing:
            return (f"no saved {predictor} estimator(s) for {', '.join(missing)} in {display_path(directory)} "
                    f"-- run `make runtime-estimators VMODE={args['mode']} DATASET={args['dataset']}`")
        return None
    if tool.endswith("_mps"):
        import torch
        return None if torch.backends.mps.is_available() else "MPS is not available on this machine"
    if tool.startswith("deepfault_"):
        port = tool[len("deepfault_"):]
        return None if keras_ok.get(port) else f"DF@{port} environment or conversion missing"
    return None


def run_one(tool: str, meta: dict, repeat: int, args: dict) -> dict:
    """One measured process. Never raises: a failure is a row with a status."""
    key = meta["config"]["key"]
    directory = ensure_dir(runs_dir(args, tool))
    record = directory / f"{key}__r{repeat}.json"
    artefact = directory / f"{key}__r{repeat}.npz"
    log_path = directory / f"{key}__r{repeat}.log"
    status_path = directory / f"{key}__r{repeat}.status.json"
    if record.exists() and not args["force"]:
        return {"tool": tool, "model": key, "repeat": repeat, "status": "cached"}

    command = command_for(tool, meta, args, record, artefact)
    started = time.perf_counter()
    status, detail = "ok", ""
    try:
        with open(log_path, "w", encoding="utf-8") as handle:
            result = subprocess.run(
                command, cwd=str(PROJECT_ROOT), stdout=handle, stderr=subprocess.STDOUT,
                timeout=args["timeout"], env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2", **EXACT_MALLOC_ENV},
            )
        if result.returncode != 0 or not record.exists():
            status = "failed"
            tail = log_path.read_text(errors="replace").strip().splitlines()[-3:]
            detail = f"exit {result.returncode}: " + " | ".join(tail)
    except subprocess.TimeoutExpired:
        status, detail = "timeout", f"exceeded {args['timeout']} s"
    except Exception as error:  # the run continues whatever one process does
        status, detail = "failed", f"{type(error).__name__}: {error}"
    wall = time.perf_counter() - started
    status_path.write_text(json.dumps({"status": status, "detail": detail, "process_wall_seconds": wall,
                                       "command": command}, indent=2) + "\n", encoding="utf-8")
    return {"tool": tool, "model": key, "repeat": repeat, "status": status, "detail": detail, "wall": wall}


def all_rows(args: dict, planned: Sequence[tuple]) -> List[dict]:
    """Every record on disk for this sample scope, plus a row for each planned run without one.

    A partial invocation (``--model``, ``--tool``) therefore never drops the
    runs of an earlier one from the table.
    """
    rows, seen = [], set()
    scope = runtime_root(args["mode"], args["dataset"]) / "runs" / run_tag(args)
    for tool in ALL_TOOLS:
        directory = scope / tool
        if not directory.exists():
            continue
        for status in sorted(directory.glob("*__r*.status.json")):
            key, _, repeat = status.name[: -len(".status.json")].rpartition("__r")
            rows.append(flatten(tool, key, int(repeat), directory))
            seen.add((tool, key, int(repeat)))
    for repeat, meta, tool in planned:
        if (tool, meta["config"]["key"], repeat) not in seen:
            rows.append(flatten(tool, meta["config"]["key"], repeat, scope / tool))
    return sorted(rows, key=lambda r: (r["tool"], r["model"], r["repeat"]))


# --------------------------------------------------------------------------
# 4 fidelity
# --------------------------------------------------------------------------


def spectrum_from_npz(path: Path, n_layers: int) -> Dict[str, List[np.ndarray]]:
    blob = np.load(path)
    out = {"a_f": [blob[f"a_f_{i}"] for i in range(n_layers)], "a_s": [blob[f"a_s_{i}"] for i in range(n_layers)]}
    for metric in METRICS:
        if f"{metric}_0" in blob.files:
            out[metric] = [blob[f"{metric}_{i}"] for i in range(n_layers)]
    return out


def fidelity_rows(metas: Sequence[dict], args: dict) -> List[dict]:
    """Every dynamic tool's spectrum vs. the score-step cache and vs. HS@pt on the CPU.

    Two references, because they answer different questions. The cache was
    counted by make score (on its device, usually MPS); a tool on another device
    can differ from it where a pre-activation lies within float rounding of the
    threshold. HS@pt on the CPU is the same arithmetic class as the
    TensorFlow ports and ICSoft@pt on the CPU. Spearman rho of the scores says
    whether a handful of off-by-one counts matters for the ranking at all.
    """
    from staticsbfl.persistence import load_suspiciousness

    rows = []
    for meta in metas:
        key = meta["config"]["key"]
        layers = len(meta["hidden_widths"])
        references = {}
        if Path(meta["spectrum"]).exists():
            bundle = load_suspiciousness(Path(meta["spectrum"]))
            names = bundle.hit_spectrum.layer_names
            references["score-step cache"] = {
                "a_f": [bundle.hit_spectrum.layers[n].a_f.flatten().numpy() for n in names],
                "a_s": [bundle.hit_spectrum.layers[n].a_s.flatten().numpy() for n in names],
                **{m: [bundle.scores[m][n].flatten().double().numpy() for n in names] for m in METRICS},
            }
        cpu = runs_dir(args, "hs_pt_cpu") / f"{key}__r0.npz"
        if cpu.exists():
            references["hs_pt_cpu"] = spectrum_from_npz(cpu, layers)
        for tool in DYNAMIC_TOOLS:
            path = runs_dir(args, tool) / f"{key}__r0.npz"
            if not path.exists():
                continue
            got = spectrum_from_npz(path, layers)
            for reference, ref in references.items():
                if reference != tool:
                    rows.append({"tool": tool, "model": key, "reference": reference, **compare_spectra(got, ref)})
    return rows


def write_csv(path: Path, rows: List[dict], columns: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in columns})


def environment() -> dict:
    """Machine, interpreter and library versions, for the evaluation-environment paragraph."""
    def run(command):
        try:
            out = subprocess.run(command, capture_output=True, text=True,
                                 env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}).stdout.strip()
            return out.splitlines()[-1] if out else "unavailable"
        except OSError:
            return "unavailable"

    import numpy
    import sklearn
    import torch

    info = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0],
            "cpu": run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
            "memory_bytes": run(["sysctl", "-n", "hw.memsize"]), "logical_cpus": os.cpu_count(),
            "memory_source": MEMORY_SOURCE, "memory_note": MEASUREMENT_NOTE,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "versions_main": f"python {sys.version.split()[0]} · torch {torch.__version__} · numpy {numpy.__version__} "
                             f"· scikit-learn {sklearn.__version__} ({sys.prefix})"}
    for port in PORTS:
        python = port_python(port)
        if python.exists():
            info[f"versions_deepfault_{port}"] = run([str(python), "-c", (
                "import sys, numpy, h5py, tensorflow as tf, keras\n"
                "try:\n import tf_keras; tk = ' · tf_keras ' + tf_keras.__version__\n"
                "except ImportError:\n tk = ''\n"
                "print('python', sys.version.split()[0], '· tensorflow', tf.__version__, '· keras', keras.__version__"
                " + tk, '· numpy', numpy.__version__, '· h5py', h5py.__version__, '(' + sys.prefix + ')')")])
    return info


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@click.command()
@click.option("--mode", type=click.Choice(sorted(POPULATION_SIZES)), default="light", show_default=True)
@click.option("--dataset", type=click.Choice(DATASETS), default="cifar10", show_default=True)
@click.option("--tool", "tools", multiple=True, type=click.Choice(ALL_TOOLS),
              help="Default: " + ", ".join(DEFAULT_TOOLS) + ".")
@click.option("--model", "keys", multiple=True, help="Restrict to these model keys.")
@click.option("--repeats", type=int, default=3, show_default=True,
              help="Fresh processes per (tool, model) for the fast tools.")
@click.option("--deepfault-repeats", type=int, default=1, show_default=True,
              help="Fresh processes per (DeepFault port, model); a run can take many minutes.")
@click.option("--samples", default="all", show_default=True,
              help="'all' = the whole test set (matches the cached spectra) or a class index "
                   "(DeepFault's own per-class protocol), the same for every dynamic tool.")
@click.option("--max-samples", type=int, default=None, help="First N test samples only (smoke runs).")
@click.option("--target-kind", type=click.Choice(("rank_model", "rank_layer", "raw")), default="rank_model",
              show_default=True, help="Which fitted target the static predictor uses.")
@click.option("--timeout", type=int, default=6 * 3600, show_default=True, help="Seconds per process.")
@click.option("--seeds-per-design", type=int, default=DEFAULT_SEEDS_PER_DESIGN, show_default=True,
              help="Seeds per designed shape (5 shapes): runtime depends on the shape, not the seed. 0 = all.")
@click.option("--force", is_flag=True, help="Redo exports and measurements already on disk.")
def main(mode, dataset, tools, keys, repeats, deepfault_repeats, samples, max_samples, target_kind, timeout,
         seeds_per_design, force):
    """Measure every tool on the phase-4 designed models, one fresh process per run."""
    configure_logging()
    root = ensure_dir(runtime_root(mode, dataset))
    add_file_log(root / "runtime.log")

    tools = tuple(tools) or DEFAULT_TOOLS
    args = {"mode": mode, "dataset": dataset, "samples": samples, "max_samples": max_samples,
            "target_kind": target_kind, "timeout": timeout, "force": force}
    phase_banner(logger, f"StaticSBFL · RUNTIME AND MEMORY · {mode} · {dataset}", [
        ("Models", [f"{PHASE} '{EVAL_SPLIT}' (established designs) -- every tool on the same models",
                    "seeds per shape: " + (str(seeds_per_design) if seeds_per_design else "all")]),
        ("Tools", [f"{t:<30}{TOOL_LABELS[t]}" for t in tools]),
        ("Samples", [f"{'whole test set' if samples == 'all' else 'class ' + samples}"
                     + (f", first {max_samples}" if max_samples else "")
                     + "  (identical input for every dynamic tool)"]),
        ("Repeats", [f"{repeats} per fast tool, {deepfault_repeats} per DeepFault port; each a fresh process"]),
        ("Memory", [style(line, "dim") for line in wrap_text(MEASUREMENT_NOTE, 76)]),
    ], outputs=[(table_path(args, "runs"), "one row per measured process"),
                (table_path(args, "fidelity"), "dynamic spectra vs. the score-step cache"),
                (root / "runtime.log", "this log")])

    env = environment()
    (root / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    logger.info("machine: %s · %s · %s logical CPUs", env["cpu"], env["platform"], env["logical_cpus"])
    for key, value in env.items():
        if key.startswith("versions_"):
            logger.info("%-22s %s", key[len("versions_"):], value)

    with Step(logger, 1, 5, "export", "test set and model weights as framework-neutral .npz",
              outputs=[runtime_test_set(dataset), root / "weights"]) as step:
        test_set = export_test_set(dataset, force)
        metas = export_models(mode, dataset, test_set, keys, force, seeds_per_design)
        if not metas:
            raise click.ClickException("No phase-4 designed models found. Run make phase PHASE=phase4 first.")
        step.summary = f"{len(metas)} models"

    ports = [t[len("deepfault_"):] for t in tools if t.startswith("deepfault_")]
    with Step(logger, 2, 5, "keras", "each DeepFault port converts the weights into its own Keras format",
              outputs=[root / f"keras_{p}" for p in ports]) as step:
        keras_ok = convert_for_ports(mode, dataset, test_set, ports, force) if ports else {}
        step.summary = ", ".join(f"DF@{p} {'ok' if v else 'FAILED'}" for p, v in keras_ok.items()) or "no port selected"

    plan, skipped = [], {}
    for tool in tools:
        reason = tool_available(tool, args, keras_ok)
        if reason:
            skipped[tool] = reason
            continue
        count = deepfault_repeats if tool.startswith("deepfault_") else repeats
        plan += [(repeat, meta, tool) for repeat in range(count) for meta in metas]
    plan.sort(key=lambda item: (item[0], item[1]["config"]["key"]))

    with Step(logger, 3, 5, "measure", ["one fresh process per (tool, model, repeat); failures are recorded",
                                        f"timeout {format_duration(timeout)} per process"],
              outputs=[root / "runs" / run_tag(args)]) as step:
        for tool, reason in skipped.items():
            logger.warning("SKIPPED %s: %s", tool, reason)
        started = time.perf_counter()
        counts: Dict[str, int] = {}
        for index, (repeat, meta, tool) in enumerate(plan, start=1):
            result = run_one(tool, meta, repeat, args)
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            if result["status"] == "cached":
                continue
            line = (f"[{index}/{len(plan)}] {tool:<28} {meta['config']['key']:<34} r{repeat}  "
                    f"{result['status']:<7} {format_duration(result['wall'])}  "
                    f"(elapsed {format_duration(time.perf_counter() - started)})")
            (logger.info if result["status"] == "ok" else logger.warning)(line)
            if result["detail"]:
                logger.warning("    %s", result["detail"][:300])
        step.summary = ", ".join(f"{v} {k}" for k, v in counts.items() if v) or "nothing planned"

    with Step(logger, 4, 5, "fidelity", "dynamic counts vs. the score-step cache and vs. HS@pt on the CPU",
              outputs=[table_path(args, "fidelity")]) as step:
        if samples != "all" or max_samples:
            logger.info("skipped: the cached spectra cover the whole test set, this run did not.")
            rows = []
        else:
            rows = fidelity_rows(metas, args)
            for row in rows:
                if not row["identical"]:
                    rho = ", ".join(f"{m} {row[f'spearman_{m}']:.6f}" for m in METRICS if row.get(f"spearman_{m}") is not None)
                    logger.info("%s %s vs %s: %d of %d neurons differ (max %d count) · rho %s", row["tool"],
                                row["model"], row["reference"], row["neurons_differing"], row["n_neurons"],
                                row["max_count_difference"], rho or "n/a")
        write_csv(table_path(args, "fidelity"), rows, FIDELITY_COLUMNS)
        identical = sum(1 for r in rows if r["identical"])
        step.summary = f"{identical}/{len(rows)} comparisons identical (rest: float rounding near the threshold, see rho)"

    with Step(logger, 5, 5, "table", "flatten every record into one CSV", outputs=[table_path(args, "runs")]) as step:
        rows = all_rows(args, plan)
        write_csv(table_path(args, "runs"), rows, run_columns())
        step.summary = f"{sum(r['status'] in ('ok', 'cached') for r in rows)} measured of {len(rows)} rows"
    plain(logger, style(f"  Next: make evaluate  (reads these CSVs, prints and renders the comparison)", "yellow"))


if __name__ == "__main__":
    main()
