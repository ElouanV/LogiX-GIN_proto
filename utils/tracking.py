"""MLflow tracking for LogiX-GIN experiments.

Every function is a no-op when mlflow is not installed or ``LOGIX_MLFLOW=0``, and a
tracking failure is reported once and never stops training.

Where things go
    tracking   ``$MLFLOW_TRACKING_URI`` if set, else ``sqlite:///<repo>/mlflow.db``.
               Several training processes at once should log through a server
               (``scripts/mlflow_server.sh``, then ``MLFLOW_TRACKING_URI=http://127.0.0.1:5055``):
               the server serialises writes that concurrent sqlite clients would race on.
    artifacts  ``$LOGIX_MLFLOW_ARTIFACTS/<experiment>/``, default ``<repo>/mlartifacts``
               (models, json, csv).

How runs are organised
    experiment   ``pipeline/<dataset>``: every trained model is one parent run (``kind=
                 pipeline``) with one nested run per stage (tag ``stage``):
                     teacher       the GIN teacher. Its training run is the teacher stage of
                                   its own pipeline; every student pipeline gets a reference
                                   child instead (params, final metrics, ``teacher_run_id``),
                                   since one teacher serves many students and is not retrained
                     distillation  the LogiX-GIN / prototype student (train_logic.py,
                                   train_proto.py)
                     finetune/<m>  a fine-tune of that student: qat, hoyer, harden
                 A pipeline is opened by each Optuna trial and each final fold
                 (utils/hps.py, utils/experiment.py), or by the training script when run
                 alone. The distillation stage writes ``mlflow_run.json`` (parent and run
                 ids) into its run directory, so a fine-tune started later from that
                 directory nests under the same parent; a directory without it gets a new
                 parent. Runs from before (2026-10-15) stay in the per-stage experiments
                 ``baseline/``, ``logic/``, ``proto/``, ``qat/``, ``harden/``, ``sparsify/<dataset>``.
    run          one per trained seed, tagged ``config`` (the results-folder string that
                 identifies the hyper-parameters), ``seed``, ``kind=seed``. Seeds of one
                 configuration can run in parallel processes; group them in the UI by
                 the ``config`` tag. ``kind=summary`` runs hold the mean/std over seeds.
    losses       every training loop logs ``train_loss`` and ``val_loss`` per epoch (the same
                 objective on the val split, without gradient: utils.evaluation.held_out).
    registry     every seed's best checkpoint becomes a new version of a registered
                 model (``gin-teacher-<ds>``, ``logix-proto-<ds>-<level>``,
                 ``logix-proto-sparse-<ds>-<level>``) tagged with its accuracy and path.

Per-epoch metrics (``log_metrics(..., step=k)``) are buffered and sent in batches every
``LOGIX_MLFLOW_FLUSH_STEPS`` steps (default 200) or ``LOGIX_MLFLOW_FLUSH_SECS`` seconds
(default 120), and when the run ends: one HTTP call per epoch took 3.4 s with a dozen
runs writing to the server, which left the training processes idle. Every step is
still recorded, with the time it was logged.

CO₂: if codecarbon is installed (and ``LOGIX_CO2`` is not 0), every run is measured with
an offline tracker (grid of ``LOGIX_CO2_COUNTRY``, default FRA) and logs ``co2/emissions_kg``,
``co2/energy_kwh`` and its GPU / CPU / RAM parts. CPU and RAM are this process's share
(tracking_mode='process'), but the GPU part is the whole GPU's energy over the run, so
runs sharing the GPU each count it: sum ``co2/gpu_kwh`` over parallel runs and it
over-counts by about the number of jobs. Report a sweep's total from the per-run
CPU/RAM sums plus the GPU energy of one run per parallel slot. A pipeline parent has no
tracker of its own: it logs the sum of the stages that ran in its process.
"""
import json
import contextlib
import os
import subprocess
import time
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import mlflow
except ImportError:                                    # tracking is optional
    mlflow = None

try:
    from codecarbon import OfflineEmissionsTracker
except ImportError:                                    # CO₂ measurement is optional too
    OfflineEmissionsTracker = None

_warned = False
_buffer = []                                           # pending per-step metrics
_last_flush = time.time()
_pipelines = []                                        # open pipeline parents of this process
PIPELINE_FILE = 'mlflow_run.json'                      # written into a distillation run directory
CO2_KEYS = ('co2/emissions_kg', 'co2/energy_kwh', 'co2/gpu_kwh', 'co2/cpu_kwh', 'co2/ram_kwh', 'co2/duration_s')


def enabled():
    return mlflow is not None and os.environ.get('LOGIX_MLFLOW', '1') != '0'


def _safe(fn):
    def wrapper(*a, **k):
        global _warned
        if not enabled():
            return None
        try:
            return fn(*a, **k)
        except Exception as e:                         # never let tracking kill a run
            if not _warned:
                warnings.warn(f'MLflow tracking failed, training continues without it: {e!r}')
                _warned = True
            return None
    return wrapper


def _git():
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                             cwd=REPO, text=True).strip())
        return {'git_commit': sha, 'git_dirty': str(dirty)}
    except Exception:
        return {}


def _set_uri():
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI',
                                           f'sqlite:///{os.path.join(REPO, "mlflow.db")}'))


@_safe
def _set_experiment(experiment):
    _set_uri()
    if mlflow.get_experiment_by_name(experiment) is None:
        root = os.environ.get('LOGIX_MLFLOW_ARTIFACTS', os.path.join(REPO, 'mlartifacts'))
        loc = 'file://' + os.path.join(root, experiment)
        try:
            mlflow.create_experiment(experiment, artifact_location=loc)
        except Exception:                              # created by a parallel process meanwhile
            pass
    mlflow.set_experiment(experiment)


def _start_co2():
    if OfflineEmissionsTracker is None or os.environ.get('LOGIX_CO2', '1') == '0':
        return None
    try:
        t = OfflineEmissionsTracker(country_iso_code=os.environ.get('LOGIX_CO2_COUNTRY', 'FRA'),
                                    save_to_file=False, log_level='error', tracking_mode='process',
                                    allow_multiple_runs=True)
        t.start()
        return t
    except Exception as e:
        warnings.warn(f'CO2 tracking could not start, training continues without it: {e!r}')
        return None


@_safe
def _stop_co2(tracker):
    """Stop the tracker and log its totals into the active run; they are also added to the
    open pipeline parent's total."""
    if tracker is None:
        return
    kg = tracker.stop()
    d = tracker.final_emissions_data
    co2 = dict(zip(CO2_KEYS, (kg, d.energy_consumed, d.gpu_energy, d.cpu_energy, d.ram_energy, d.duration)))
    log_metrics(co2)
    if _pipelines:
        total = _pipelines[-1]['co2']
        for k, v in co2.items():
            total[k] = total.get(k, 0.0) + (v or 0.0)


def pipeline_experiment(dataset):
    return f'pipeline/{dataset}'


@contextlib.contextmanager
def pipeline(dataset, run_name, params=None, tags=None):
    """Open the parent run of one trained model's stages (yields the run or None). Stages
    opened inside the block with ``run(..., stage=...)`` nest under it."""
    active = None
    if enabled():
        try:
            _set_experiment(pipeline_experiment(dataset))
            flush()
            active = mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None,
                                      tags=_tag_strings({**_git(), 'dataset': dataset, **(tags or {}), 'kind': 'pipeline',
                                                         'pipeline_kind': (tags or {}).get('kind')}))
            log_params(params or {})
        except Exception as e:
            warnings.warn(f'MLflow pipeline run could not start, training continues without it: {e!r}')
            active = None
    entry = {'run': active, 'dataset': dataset, 'co2': {}, 'teacher': False}
    _pipelines.append(entry)
    status = 'FINISHED'
    try:
        yield active
    except BaseException:
        status = 'FAILED'
        raise
    finally:
        _pipelines.remove(entry)
        if active is not None:
            log_metrics(entry['co2'])
            _end(status)


def in_pipeline():
    """True inside a ``pipeline`` block of this process."""
    return bool(_pipelines)


def _tag_strings(tags):
    # the REST store (tracking server) rejects non-string tag values; sqlite does not
    return {k: str(v) for k, v in tags.items() if v is not None}


def read_pipeline(run_dir):
    """The ids a distillation stage wrote into ``run_dir`` (``PIPELINE_FILE``), or None."""
    f = os.path.join(run_dir, PIPELINE_FILE)
    if not os.path.exists(f):
        return None
    try:
        return json.load(open(f))
    except Exception:
        return None


@_safe
def write_pipeline(run_dir):
    """Record the active stage run and its parent in ``run_dir``, so later fine-tunes of
    that directory nest under the same parent."""
    r = mlflow.active_run()
    if r is None or not os.path.isdir(run_dir):
        return
    with open(os.path.join(run_dir, PIPELINE_FILE), 'w') as f:
        json.dump({'run_id': r.info.run_id, 'parent_run_id': r.data.tags.get('mlflow.parentRunId'),
                   'experiment_id': r.info.experiment_id, 'stage': r.data.tags.get('stage')}, f)


@contextlib.contextmanager
def run(experiment, run_name, params=None, tags=None, stage=None, dataset=None, parent_run_id=None, co2=True):
    """Open an MLflow run for the duration of the block (yields the run or None).

    With ``stage`` (teacher, distillation, finetune/<method>) the run is a nested stage of
    a pipeline (experiment ``pipeline/<dataset>``, ``experiment`` kept as the tag
    ``stage_experiment``): of ``parent_run_id`` if given (a finished parent, e.g. read from
    a run directory's ``mlflow_run.json``), else of the open pipeline of this process, else
    of a new pipeline named ``run_name``. Without ``stage``: a plain run of ``experiment``.
    ``co2=False``: no CO₂ tracker (runs that only copy metrics).
    """
    if stage is not None and parent_run_id is None and not _pipelines:
        with pipeline(dataset, run_name, tags=tags):
            with run(experiment, run_name, params, tags, stage, dataset, co2=co2) as r:
                yield r
        return
    active = None
    tracker = None
    if enabled():
        try:
            tags = {**_git(), **(tags or {})}
            if stage is None:
                _set_experiment(experiment)
                active = mlflow.start_run(run_name=run_name, tags=_tag_strings(tags))
            else:
                tags.update({'stage': stage, 'stage_experiment': experiment, 'dataset': dataset})
                _set_experiment(pipeline_experiment(dataset))
                flush()                                # the parent's pending steps
                if parent_run_id is not None and (mlflow.active_run() is None
                                                  or mlflow.active_run().info.run_id != parent_run_id):
                    active = mlflow.start_run(run_name=f'{stage}', parent_run_id=parent_run_id,
                                              tags=_tag_strings(tags))
                else:
                    active = mlflow.start_run(run_name=f'{stage}', nested=True, tags=_tag_strings(tags))
            log_params(params or {})
            tracker = _start_co2() if co2 else None
        except Exception as e:
            warnings.warn(f'MLflow run could not start, training continues without it: {e!r}')
            active = None
    try:
        yield active
    except BaseException:
        if active is not None:
            _stop_co2(tracker)
            _end('FAILED')
            active = None
        raise
    finally:
        if active is not None:
            _stop_co2(tracker)
            _end('FINISHED')


def ensure_pipeline(dataset, run_name, params=None, tags=None):
    """``pipeline(...)`` unless this process already has one open (then a no-op block)."""
    return contextlib.nullcontext() if _pipelines else pipeline(dataset, run_name, params, tags)


def fine_tune_parent(run_dir):
    """Parent of a fine-tune of ``run_dir``: None inside an open pipeline (the stage nests
    there), else the parent recorded in the directory (None: a new pipeline is opened)."""
    if _pipelines:
        return None
    rec = read_pipeline(run_dir)
    return rec.get('parent_run_id') if rec else None


@_safe
def _teacher_source(teacher_dir, dataset):
    """Run id of a teacher's training run: from its directory, else the latest seed run of
    the legacy experiment baseline/<dataset> with the same config and seed."""
    rec = read_pipeline(teacher_dir)
    if rec:
        return rec['run_id']
    cfg_dir, seed = os.path.split(os.path.normpath(teacher_dir))
    cfg = os.path.basename(cfg_dir)
    exp = mlflow.get_experiment_by_name(f'baseline/{dataset}')
    if exp is None:
        return None
    runs = mlflow.MlflowClient().search_runs(
        [exp.experiment_id], f"tags.config = '{cfg}' and tags.seed = '{seed}' and tags.kind = 'seed'",
        order_by=['attributes.start_time DESC'], max_results=1)
    return runs[0].info.run_id if runs else None


def teacher_reference(teacher_dir, dataset, seed):
    """The teacher stage of the open pipeline, once: a reference to the teacher's training
    run (its parameters, final metrics, ``teacher_run_id``), which is not repeated."""
    if not enabled() or not _pipelines or _pipelines[-1]['teacher']:
        return
    _pipelines[-1]['teacher'] = True
    _set_uri()
    src = _teacher_source(teacher_dir, dataset)
    metrics = {}
    if src:
        try:
            metrics = {k: v for k, v in mlflow.MlflowClient().get_run(src).data.metrics.items()
                       if k.startswith(('final/', 'best_', 'co2/'))}
        except Exception:
            metrics = {}
    args_file = os.path.join(teacher_dir, 'args.json')
    params = json.load(open(args_file)) if os.path.exists(args_file) else {}
    with run(f'baseline/{dataset}', 'teacher', params={**params, 'seed': seed},
             tags={'kind': 'reference', 'teacher_run_id': src, 'teacher_path': teacher_dir, 'seed': seed},
             stage='teacher', dataset=dataset, co2=False):
        log_metrics({k: v for k, v in metrics.items() if not k.startswith('co2/')})
        # the teacher's energy was spent once, by its own run: not added to this pipeline's total
        log_metrics({f'teacher_{k}': v for k, v in metrics.items() if k.startswith('co2/')})


@_safe
def _end(status):
    flush()
    mlflow.end_run(status=status)


@_safe
def flush():
    """Send the buffered per-step metrics of the active run."""
    global _last_flush
    _last_flush = time.time()
    if not _buffer or not _active():
        _buffer.clear()
        return
    from mlflow.entities import Metric
    run_id = mlflow.active_run().info.run_id
    metrics = [Metric(k, v, ts, step) for k, v, ts, step in _buffer]
    _buffer.clear()
    client = mlflow.MlflowClient()
    for i in range(0, len(metrics), 1000):             # log_batch limit
        client.log_batch(run_id, metrics=metrics[i:i + 1000])


def _active():
    return enabled() and mlflow.active_run() is not None


@_safe
def log_params(params):
    if _active():
        # values are stringified by mlflow; keep them short
        mlflow.log_params({k: (str(v)[:500]) for k, v in params.items()})


@_safe
def log_metrics(metrics, step=None):
    if _active():
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        if step is None:                               # final values: send now
            flush()
            mlflow.log_metrics(clean)
            return
        ts = int(time.time() * 1000)
        _buffer.extend((k, v, ts, int(step)) for k, v in clean.items())
        steps = len({b[3] for b in _buffer})
        if (steps >= int(os.environ.get('LOGIX_MLFLOW_FLUSH_STEPS', 200))
                or time.time() - _last_flush >= float(os.environ.get('LOGIX_MLFLOW_FLUSH_SECS', 120))):
            flush()


@_safe
def set_tags(tags):
    if _active():
        mlflow.set_tags({k: str(v) for k, v in tags.items()})


@_safe
def log_artifact(path, artifact_path=None):
    if _active() and os.path.exists(path):
        mlflow.log_artifact(path, artifact_path=artifact_path)


@_safe
def log_dict(d, name):
    if _active():
        mlflow.log_dict(d, name)


@_safe
def log_model(model, registered_name, code_dirs=(), tags=None):
    """Log a torch model and register it as a new version of ``registered_name``.

    ``code_dirs`` (relative to the repo) are shipped with the model so the registered
    version loads without this checkout: ``mlflow.pytorch.load_model(<uri>)``.
    """
    if not _active():
        return None
    import mlflow.pytorch
    kw = dict(registered_model_name=registered_name,
              code_paths=[os.path.join(REPO, d) for d in code_dirs],
              pip_requirements=['torch', 'torch_geometric'])
    try:
        # mlflow >= 3.x defaults to 'pt2', a traced graph that needs an example input and
        # cannot follow a GNN's variable-size batches; 'pickle' stores the module like torch.save
        info = mlflow.pytorch.log_model(model, name='model', serialization_format='pickle', **kw)
    except TypeError:                                                      # mlflow 2.x
        info = mlflow.pytorch.log_model(model, artifact_path='model', **kw)
    version = getattr(info, 'registered_model_version', None)
    if tags and version is not None:
        client = mlflow.MlflowClient()
        for k, v in tags.items():
            client.set_model_version_tag(registered_name, str(version), k, str(v))
    return info


@_safe
def log_to_run_id(run_id, metrics, artifacts=()):
    """Log metrics (and files) into an existing, finished run."""
    from mlflow.entities import Metric
    _set_uri()
    client = mlflow.MlflowClient()
    ts = int(time.time() * 1000)
    clean = [Metric(k, float(v), ts, 0) for k, v in metrics.items() if v is not None]
    for i in range(0, len(clean), 1000):
        client.log_batch(run_id, metrics=clean[i:i + 1000])
    for a in artifacts:
        if os.path.exists(a):
            client.log_artifact(run_id, a)
    return run_id


@_safe
def log_to_run(experiments, tags, metrics, artifacts=()):
    """Log metrics (and files) into an existing, finished run: the latest run of the given
    experiments carrying all ``tags`` (e.g. a final fold's training run, to add the
    metrics computed afterwards from its checkpoint). Returns its run id, None if none."""
    _set_uri()
    client = mlflow.MlflowClient()
    ids = [e.experiment_id for e in map(mlflow.get_experiment_by_name, experiments) if e is not None]
    if not ids:
        return None
    filt = ' and '.join(f"tags.{k} = '{v}'" for k, v in tags.items())
    runs = client.search_runs(ids, filt, order_by=['attributes.start_time DESC'], max_results=1)
    if not runs:
        return None
    return log_to_run_id(runs[0].info.run_id, metrics, artifacts)
