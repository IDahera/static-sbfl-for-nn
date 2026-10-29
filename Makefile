# StaticSBFL -- task runner. `make help` lists everything.
#
# The study runs four phases. Each phase is four steps, in this order:
#
#   1. train     generate the phase's architectures, train and cache every model
#   2. score     hit spectrum per model -> Tarantula / Ochiai / D* per neuron
#   3. neurons   the neuron table: static features + measured targets
#   4. predict   fit the predictors, rank held-out neurons, compare with measured
#
# `make light` and `make heavy` run all four phases in one budget; `make runtime`
# compares the static prediction with the dynamic tools on the phase-4 models;
# `make evaluate` summarises whatever exists. `make all` does all of it.
#
# Every step skips work that is already cached, so an interrupted run resumes
# where it stopped. Add FORCE=--force to redo cached work anyway.

# Allow local, uncommitted developer overrides (e.g. PYTHON=...)
-include config.mk

# A bare `make` prints the help, never starts an hours-long run.
.DEFAULT_GOAL := help

# Detect Python interpreter:
# 1. Explicit override via command-line or config.mk (handled above)
# 2. Active Conda or virtualenv ($CONDA_PREFIX or $VIRTUAL_ENV)
# 3. Local repo .venv if it exists
# 4. System python3
ifeq ($(origin PYTHON), undefined)
  ifneq ($(CONDA_PREFIX),)
    PYTHON := python
  else ifneq ($(VIRTUAL_ENV),)
    PYTHON := python
  else ifneq ($(wildcard .venv/bin/python),)
    PYTHON := .venv/bin/python
  else
    PYTHON := python3
  endif
endif
DEVICE  ?= auto
FORCE   ?=
VMODE   ?= light
METRIC  ?=
PHASE   ?= phase1
DATASET ?= cifar10

.PHONY: help venv install test \
        train score neurons predict preview phase \
        light heavy debug status evaluate \
        runtime-setup costs runtime-estimators runtime smoke all trim cache-size \
        paper-phase1 paper-phase2 paper-phase3 paper-runtime \
        clean-variants clean-variants-all clean-debug clean-runtime clean

help:
	@echo "Setup"
	@echo "  make install           create .venv and install the package (editable)"
	@echo "  make test              run the unit test suite"
	@echo ""
	@echo "The four phases"
	@echo "  make light                              all four phases, 200 fit / 40 eval models, 3 epochs"
	@echo "  make heavy                              all four phases, 400 fit / 80 eval models, 10 epochs"
	@echo "  make debug                              pipeline check: 20 / 4 models, 1 epoch, cache/variants-debug/"
	@echo "  make smoke                              debug + CPU runtime (no DeepFault) + evaluate: every result type in minutes"
	@echo "  make paper-phase1|2|3, paper-runtime    the paper's tables, one target per table (README)"
	@echo "  make status                             what is done / missing, time and disk"
	@echo "  make preview PHASE=phase2               list the population, build nothing"
	@echo "  make phase   PHASE=phase1               one phase end to end"
	@echo "  make train / score / neurons / predict  the four steps of one phase"
	@echo "  Variables: PHASE=$(PHASE)  VMODE=$(VMODE) (debug|light|heavy)  DATASET=$(DATASET) (cifar10|fashion_mnist)"
	@echo "             METRIC='--metric ochiai' narrows the predictor sweep"
	@echo ""
	@echo "Runtime and memory (phase-4 designed models)"
	@echo "  make runtime-setup      the two DeepFault conda envs (deepfault-tf1, deepfault-tf2)"
	@echo "  make runtime            StaticSBFL vs HS@pt (cpu, mps) vs DF@tf1 vs DF@tf2, light + heavy"
	@echo "  make runtime-estimators refit phase 4 and save the predictors the runtime run loads"
	@echo "  make costs              training / score / fit time of the whole run, light and heavy"
	@echo "  Variables: RUNTIME_ARGS='--max-samples 500 --repeats 1'  RUNTIME_ARGS='--tool icsoft_pt_cpu'"
	@echo ""
	@echo "Results"
	@echo "  make evaluate           read-only: everything that exists, logged + cache/evaluation/<dataset>/report.html"
	@echo "  make all                light -> heavy -> runtime -> evaluate"
	@echo "  Variables: SCOPE=$(SCOPE) (active|all)  EVAL_ARGS='--model-type tree_ensemble --metric ochiai'"
	@echo ""
	@echo "Disk"
	@echo "  make cache-size                         disk footprint of every part of cache/"
	@echo "  make trim VMODE=heavy                   remove models a shrunk population no longer uses (APPLY=--apply)"
	@echo "  make clean-variants PHASE=phase1 VMODE=light DATASET=cifar10   one phase (asks first)"
	@echo "  make clean-variants-all                 every phase, light AND heavy (asks first)"
	@echo "  make clean-debug                        cache/variants-debug/ (asks first)"
	@echo "  make clean-runtime                      cache/runtime/ (asks first)"
	@echo "  make clean                              delete cache/ (asks first, defaults to no)"
	@echo ""
	@echo "Variables: DEVICE=$(DEVICE) (auto|cpu|mps|cuda)  FORCE=$(FORCE)"

venv:
	@test -d .venv || python3 -m venv .venv

install:
ifeq ($(PYTHON),.venv/bin/python)
	@test -d .venv || $(MAKE) venv
endif
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

# --- the four steps of one phase ---------------------------------------------
# Four phases, each fitting a predictor on one population and evaluating it on
# others. PHASE=phase1..phase4; DATASET=cifar10|fashion_mnist.
#
#   phase1  reseeding only          the noise floor
#   phase2  held-out architectures  same region
#   phase3  one axis at a time      deeper / wider / saturating
#   phase4  sampled -> designed     transfer to standard geometries
#
# Extra arguments for train only, e.g. TRAIN_ARGS=--retrain-mismatched to
# retrain just the cached models trained under another epoch count.
TRAIN_ARGS ?=

train:
	$(PYTHON) -m staticsbfl.cli.phase_train --phase $(PHASE) --mode $(VMODE) --dataset $(DATASET) --device $(DEVICE) $(FORCE) $(TRAIN_ARGS)

score:
	$(PYTHON) -m staticsbfl.cli.phase_score --phase $(PHASE) --mode $(VMODE) --dataset $(DATASET) --device $(DEVICE) $(FORCE)

neurons:
	$(PYTHON) -m staticsbfl.cli.phase_neurons --phase $(PHASE) --mode $(VMODE) --dataset $(DATASET)

# Every suspiciousness metric by default -- a predictor of Tarantula is not a
# predictor of D*. Narrow it with METRIC='--metric ochiai' when iterating.
# Phase 4 also keeps its gradient-boosting and MLP estimators: the runtime
# comparison loads them to time the static prediction.
# Only the rank_model estimators are kept -- the runtime comparison uses no other.
PREDICT_SAVE = $(if $(filter phase4,$(PHASE)),--save-predictor gradient_boosting --save-predictor mlp --save-kind rank_model)

predict:
	$(PYTHON) -m staticsbfl.cli.phase_predict --phase $(PHASE) --mode $(VMODE) --dataset $(DATASET) $(METRIC) $(PREDICT_SAVE)

preview:
	@$(PYTHON) -m staticsbfl.cli.phase_train --phase $(PHASE) --mode $(VMODE) --dataset $(DATASET) --dry-run

# One phase, end to end: a banner saying what the phase asks, which files hold
# its result and how long it will take; the four steps; a closing check of what
# was written. GLOSSARY=--no-glossary drops the metric legend from the banner,
# which light / heavy do after printing it once at the top.
GLOSSARY ?= --glossary
REPORT = $(PYTHON) -m staticsbfl.cli.status

phase:
	@$(REPORT) start --phase $(PHASE) --mode $(VMODE) --dataset $(DATASET) $(GLOSSARY)
	@started=$$(date +%s); \
	$(MAKE) --no-print-directory train   PHASE=$(PHASE) VMODE=$(VMODE) DATASET=$(DATASET) && \
	$(MAKE) --no-print-directory score   PHASE=$(PHASE) VMODE=$(VMODE) DATASET=$(DATASET) && \
	$(MAKE) --no-print-directory neurons PHASE=$(PHASE) VMODE=$(VMODE) DATASET=$(DATASET) && \
	$(MAKE) --no-print-directory predict PHASE=$(PHASE) VMODE=$(VMODE) DATASET=$(DATASET); \
	status=$$?; \
	$(REPORT) end --phase $(PHASE) --mode $(VMODE) --dataset $(DATASET) --since $$started; \
	exit $$status

# --- all four phases ------------------------------------------------------------
# DATASET=fashion_mnist runs the control population, which lands in its own
# cache tree and leaves CIFAR-10 untouched. Light, heavy and debug have separate
# trees as well (cache/variants/, cache/variants-heavy/, cache/variants-debug/):
# a heavy run trains every one of its models under the heavy budget instead of
# reusing 3-epoch light checkpoints.
PHASES = phase1 phase2 phase3 phase4

light heavy:
	@$(REPORT) overview --mode $@ --dataset $(DATASET)
	@for phase in $(PHASES); do \
	  $(MAKE) --no-print-directory phase PHASE=$$phase VMODE=$@ DATASET=$(DATASET) GLOSSARY=--no-glossary || exit 1; \
	done

# A pipeline check, not a result: 20 fit / 4 eval models per population, one
# epoch, and its own cache tree (cache/variants-debug/), so it can never touch
# a real population. Use it after changing the code to confirm that every phase
# runs and writes the files it should; do not read the numbers it produces.
debug:
	@echo "=== debug: 20 fit / 4 eval models, 1 epoch, cache/variants-debug/ ==="
	@for phase in $(PHASES); do \
	  $(MAKE) --no-print-directory phase PHASE=$$phase VMODE=debug DATASET=$(DATASET) GLOSSARY=--no-glossary || exit 1; \
	done

# What is done, what is missing, how long the rest would take and how much disk
# it needs, for every phase. Runs no model and writes nothing.
# Light and heavy, both datasets, unless VMODE / DATASET is given explicitly.
status:
	@$(REPORT) status $(if $(filter command line,$(origin VMODE)),--mode $(VMODE)) \
	  $(if $(filter command line,$(origin DATASET)),--dataset $(DATASET))

# The final evaluation, read only: legend, best case per phase and population
# with its baselines, overview, light vs heavy, caveats, costs, the runtime
# comparison -- printed, logged to cache/evaluation/<dataset>/evaluation.log,
# and every row as cache/evaluation/<dataset>/report.html. Light and heavy by
# default (whichever exists); VMODE=light on the command line narrows it.
# SCOPE=all puts the headlines on every neuron instead of the active ones;
# EVAL_ARGS='--model-type tree_ensemble --metric ochiai' filters.
SCOPE ?= active
EVAL_ARGS ?=
evaluate:
	@$(PYTHON) -m staticsbfl.cli.evaluate $(if $(filter command line,$(origin VMODE)),--mode $(VMODE)) \
	  --dataset $(DATASET) --scope $(SCOPE) $(EVAL_ARGS)

# --- runtime and memory ---------------------------------------------------------
# Every tool is applied to the 20 designed models of phase 4, one fresh process
# per (tool, model, repeat): StaticSBFL (the phase-4 predictors), HS@pt (this
# repository's streaming spectrum), ICSoft@pt (opt-in), both on CPU and MPS,
# and the two DeepFault ports in their own conda environments
# (baselines/deepfault_tf1, baselines/deepfault_tf2), built on the same Python
# build as $(PYTHON).
RUNTIME_ARGS ?=
CONDA ?= $(or $(CONDA_EXE),$(shell command -v conda 2>/dev/null),conda)
# The exact interpreter build of the main environment, so all tools share it.
PY_BUILD = python=3.14.2=h1858270_101_cp314

runtime-setup:
	@for port in tf1 tf2; do \
	  if ! $(CONDA) env list | grep -q "^deepfault-$$port "; then \
	    $(CONDA) create -y -q -n deepfault-$$port "$(PY_BUILD)" pip -c defaults --override-channels || exit 1; \
	  fi; \
	  $(CONDA) run -n deepfault-$$port python -m pip install -q -r baselines/deepfault_$$port/requirements.txt || exit 1; \
	done
	$(CONDA) run -n deepfault-tf1 python -m pip install -q --no-deps -r baselines/deepfault_tf1/requirements-nodeps.txt

costs:
	@$(PYTHON) -m staticsbfl.cli.costs

runtime-estimators:
	$(PYTHON) -m staticsbfl.cli.phase_predict --phase phase4 --mode $(VMODE) --dataset $(DATASET) \
	  --save-predictor gradient_boosting --save-predictor mlp --save-kind rank_model

# caffeinate (macOS) keeps the machine awake for the hours DeepFault needs; a
# sleeping machine would stretch the wall clock of whatever process is running.
KEEP_AWAKE = $(if $(shell command -v caffeinate 2>/dev/null),caffeinate -i)

# One runtime run per mode that has phase-4 models (light and heavy unless VMODE
# is given on the command line); the phase-4 estimators are refitted first if
# they were never saved. Resumable: finished measurements are skipped.
RUNTIME_MODES = $(if $(filter command line,$(origin VMODE)),$(VMODE),light heavy)

runtime:
	@for mode in $(RUNTIME_MODES); do \
	  case $$mode in heavy) tree=cache/variants-heavy ;; debug) tree=cache/variants-debug ;; *) tree=cache/variants ;; esac; \
	  tree=$$tree/$(DATASET); \
	  if [ -z "$$(ls $$tree/phase4/designed/*.pt 2>/dev/null)" ]; then \
	    echo "runtime: no phase-4 models for $$mode yet -- skipped (make $$mode first)"; continue; fi; \
	  if [ -z "$$(ls $$tree/phase4/predictions/estimators/*gradient_boosting.joblib 2>/dev/null)" ]; then \
	    $(MAKE) --no-print-directory runtime-estimators VMODE=$$mode DATASET=$(DATASET) || exit 1; fi; \
	  $(KEEP_AWAKE) $(PYTHON) -m staticsbfl.cli.runtime_benchmark --mode $$mode --dataset $(DATASET) $(RUNTIME_ARGS) || exit 1; \
	done

# The paper's results, one target per table (see README, "Reproduce the paper").
# PAPER_RUNTIME_TOOLS empty = every tool, DeepFault included (needs make
# runtime-setup); the Docker images set it to the tools that run in a container.
PAPER_RUNTIME_TOOLS ?=
paper-phase1:
	@$(MAKE) --no-print-directory phase PHASE=phase1 VMODE=light DATASET=$(DATASET)
	@$(MAKE) --no-print-directory phase PHASE=phase1 VMODE=heavy DATASET=$(DATASET)
	@$(MAKE) --no-print-directory evaluate SCOPE=all DATASET=$(DATASET) EVAL_ARGS='--phase phase1'
paper-phase2:
	@$(MAKE) --no-print-directory phase PHASE=phase2 VMODE=heavy DATASET=$(DATASET)
	@$(MAKE) --no-print-directory evaluate VMODE=heavy SCOPE=all DATASET=$(DATASET) EVAL_ARGS='--phase phase2'
paper-phase3:
	@$(MAKE) --no-print-directory phase PHASE=phase3 VMODE=heavy DATASET=$(DATASET)
	@$(MAKE) --no-print-directory evaluate VMODE=heavy SCOPE=all DATASET=$(DATASET) EVAL_ARGS='--phase phase3'
paper-runtime:
	@$(MAKE) --no-print-directory phase PHASE=phase4 VMODE=heavy DATASET=$(DATASET)
	@$(MAKE) --no-print-directory runtime VMODE=heavy DATASET=$(DATASET) RUNTIME_ARGS='$(PAPER_RUNTIME_TOOLS)'
	@$(MAKE) --no-print-directory costs
	@$(MAKE) --no-print-directory evaluate VMODE=heavy SCOPE=all DATASET=$(DATASET) EVAL_ARGS='--phase phase4'

# A quick end-to-end check of every result type (minutes, not hours): the debug
# populations of all four phases, a runtime run of StaticSBFL and HS@pt on the
# CPU (no DeepFault, which needs its own conda environments), and the
# evaluation. Its numbers only show that everything runs; do not read them.
SMOKE_TOOLS = --tool staticsbfl_gradient_boosting --tool staticsbfl_mlp --tool hs_pt_cpu
smoke:
	@$(MAKE) --no-print-directory debug DATASET=$(DATASET)
	@$(MAKE) --no-print-directory runtime VMODE=debug DATASET=$(DATASET) \
	  RUNTIME_ARGS='$(SMOKE_TOOLS) --repeats 1'
	@$(MAKE) --no-print-directory evaluate VMODE=debug DATASET=$(DATASET)

# The whole study, one step after the other; each step can also be run on its
# own, and evaluate shows whatever exists, with placeholders for the rest.
all:
	@$(MAKE) --no-print-directory light DATASET=$(DATASET)
	@$(MAKE) --no-print-directory heavy DATASET=$(DATASET)
	@$(MAKE) --no-print-directory runtime DATASET=$(DATASET)
	@$(MAKE) --no-print-directory evaluate DATASET=$(DATASET)

# After a population shrank: remove what the smaller sizes no longer use. Dry
# run by default; APPLY=--apply deletes. Kept models are never touched.
APPLY ?=
trim:
	$(PYTHON) -m staticsbfl.cli.trim --mode $(VMODE) --dataset $(DATASET) $(APPLY)

# --- disk footprint -------------------------------------------------------------
# What every part of cache/ occupies, and which target deletes it. Reads file
# sizes only, so it is safe next to a running experiment.
cache-size:
	@$(PYTHON) -m staticsbfl.cli.cache_size

# --- cleaning -------------------------------------------------------------------
# Every destructive target prints the size of what it is about to delete before
# it asks. `$(SIZE_OF) path ...` lists each path that exists, then the total.
SIZE_OF = sh -c 'set -- $$(for p in "$$@"; do [ -e "$$p" ] && echo "$$p"; done); \
  if [ $$\# -eq 0 ]; then echo "    (nothing there -- nothing would be deleted)"; \
  else du -sh "$$@" | sed "s/^/    /"; \
    if [ $$\# -gt 1 ]; then du -csh "$$@" | tail -1 | sed "s/^/    /"; fi; fi' size-of
# Destructive targets confirm first and default to no. Everything under cache/
# is reproducible, but retraining the heavy models is not cheap.

# Validate PHASE before constructing a path so this target can never expand to
# cache/variants/ itself or to a path outside it.
# VMODE picks the tree: light -> cache/variants/, heavy -> cache/variants-heavy/,
# debug -> cache/variants-debug/. Each mode has its own, so cleaning one never
# touches a population trained under another budget.
clean-variants:
	@case "$(PHASE)" in \
	  phase1|phase2|phase3|phase4) : ;; \
	  *) echo "PHASE must be one of: phase1 phase2 phase3 phase4" >&2; exit 2 ;; \
	esac; \
	case "$(VMODE)" in \
	  light) root="cache/variants" ;; \
	  heavy) root="cache/variants-heavy" ;; \
	  debug) root="cache/variants-debug" ;; \
	  *) echo "VMODE must be one of: debug light heavy" >&2; exit 2 ;; \
	esac; \
	case "$(DATASET)" in \
	  cifar10|fashion_mnist) target="$$root/$(DATASET)/$(PHASE)" ;; \
	  *) echo "DATASET must be one of: cifar10 fashion_mnist" >&2; exit 2 ;; \
	esac; \
	echo "This deletes every cached model, spectrum and table for $(VMODE) $(DATASET)/$(PHASE)."; \
	$(SIZE_OF) "$$target"; \
	printf "Delete $$target/? [y/N] "; \
	read answer; \
	case "$$answer" in \
	  [yY]|[yY][eE][sS]) rm -rf "$$target" && echo "$$target/ deleted." ;; \
	  *) echo "Kept. Nothing was deleted." ;; \
	esac

clean-variants-all:
	@echo "This deletes all four phases, light AND heavy:"; \
	echo "  cache/variants/  cache/variants-heavy/"; \
	$(SIZE_OF) cache/variants cache/variants-heavy; \
	printf "Delete both? [y/N] "; \
	read answer; \
	case "$$answer" in \
	  [yY]|[yY][eE][sS]) rm -rf cache/variants cache/variants-heavy \
	    && echo "cache/variants/ and cache/variants-heavy/ deleted." ;; \
	  *) echo "Kept. Nothing was deleted." ;; \
	esac

clean-debug:
	@echo "This deletes the debug tree cache/variants-debug/. No real result lives in it."; \
	$(SIZE_OF) cache/variants-debug; \
	printf "Delete it? [y/N] "; \
	read answer; \
	case "$$answer" in \
	  [yY]|[yY][eE][sS]) rm -rf cache/variants-debug && echo "Debug tree deleted." ;; \
	  *) echo "Kept. Nothing was deleted." ;; \
	esac

clean-runtime:
	@echo "This deletes cache/runtime/: exported models, Keras conversions, every"; \
	echo "runtime record and the cost report. The populations are kept."; \
	$(SIZE_OF) cache/runtime; \
	printf "Delete cache/runtime/? [y/N] "; \
	read answer; \
	case "$$answer" in \
	  [yY]|[yY][eE][sS]) rm -rf cache/runtime && echo "cache/runtime/ deleted." ;; \
	  *) echo "Kept. Nothing was deleted." ;; \
	esac

clean:
	@if [ ! -d cache ]; then echo "cache/ does not exist -- nothing to delete."; exit 0; fi; \
	echo "This deletes cache/ entirely -- every dataset download, model, spectrum,"; \
	echo "table and runtime record. All of it is reproducible, but not quickly:"; \
	$(PYTHON) -m staticsbfl.cli.cache_size 2>/dev/null || $(SIZE_OF) cache/*; \
	printf "Delete cache/? [y/N] "; \
	read answer; \
	case "$$answer" in \
	  [yY]|[yY][eE][sS]) rm -rf cache && echo "cache/ deleted." ;; \
	  *) echo "Kept. Nothing was deleted." ;; \
	esac
