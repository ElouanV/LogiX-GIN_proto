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
    experiment   ``<stage>/<dataset>``, stage in {baseline, proto, sparsify}.
    run          one per trained seed, tagged ``config`` (the results-folder string that
                 identifies the hyper-parameters), ``seed``, ``kind=seed``. Seeds of one
                 configuration can run in parallel processes; group them in the UI by
                 the ``config`` tag. ``kind=summary`` runs hold the mean/std over seeds.
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
CPU/RAM sums plus the GPU energy of one run per parallel slot.
"""
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


@_safe
def _set_experiment(experiment):
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI',
                                           f'sqlite:///{os.path.join(REPO, "mlflow.db")}'))
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
    """Stop the tracker and log its totals into the active run."""
    if tracker is None:
        return
    kg = tracker.stop()
    d = tracker.final_emissions_data
    log_metrics({'co2/emissions_kg': kg, 'co2/energy_kwh': d.energy_consumed, 'co2/gpu_kwh': d.gpu_energy,
                 'co2/cpu_kwh': d.cpu_energy, 'co2/ram_kwh': d.ram_energy, 'co2/duration_s': d.duration})


@contextlib.contextmanager
def run(experiment, run_name, params=None, tags=None):
    """Open an MLflow run for the duration of the block (yields the run or None)."""
    active = None
    co2 = None
    if enabled():
        try:
            _set_experiment(experiment)
            # the REST store (tracking server) rejects non-string tag values; sqlite does not
            active = mlflow.start_run(run_name=run_name,
                                      tags={k: str(v) for k, v in {**_git(), **(tags or {})}.items()})
            log_params(params or {})
            co2 = _start_co2()
        except Exception as e:
            warnings.warn(f'MLflow run could not start, training continues without it: {e!r}')
            active = None
    try:
        yield active
    except BaseException:
        if active is not None:
            _stop_co2(co2)
            _end('FAILED')
            active = None
        raise
    finally:
        if active is not None:
            _stop_co2(co2)
            _end('FINISHED')


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
