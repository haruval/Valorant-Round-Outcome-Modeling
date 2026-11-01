# ML validation

This is the historical **rolling-multimodal-v1** run (365 channels). The current
source uses **rolling-gameplay-v2** (222 channels), whose separate bounded
validation completed October 31, 2026.
Do not rerun the commands below expecting identical features or models. Existing
v1 prediction JSON remains viewable with its frozen schema; the new runtime
rejects v1 artifacts. See the [v2 execution results](controlled-experiment.md#completed-run).

Prepared and approved October 29, 2026. The owner approved the reviewed dependency,
test, three-epoch training, local artifact reload, and preview plan. That run is
complete. The five-second trade candidate rule remains unchanged. Additional
dependencies or a larger experiment are outside this validation run.

## Completed run

All 18 runtime wheel hashes matched the reviewed pins; pip 26.2.2 was bootstrapped
without executing the older bundled installer. `pip check`, all 50 standard-library
tests, and both rolling ML tests passed on CPython 3.12.13 / macOS arm64.

The run fit the three baselines and trained the 3,746-parameter CNN for three
epochs. It used 286 windows from 23 Breeze rounds for fitting and 269 windows
from 21 Ascent rounds for validation. There is no test partition. Validation
metrics average within each round and then across rounds; lower is better:

| Model | Round-mean log loss | Round-mean Brier score |
| --- | ---: | ---: |
| Training prior | 0.6648 | 0.2359 |
| Snapshot logistic | 0.5699 | 0.1687 |
| Window-summary logistic | 0.5494 | 0.1783 |
| Temporal CNN | 0.6306 | 0.2195 |

CNN validation log loss was 0.6672, 0.6488, and 0.6306 over the three epochs;
epoch 3 was selected. The CNN did not outperform either logistic baseline in
this run. The history baseline had the lowest log loss; the snapshot baseline
had the lowest Brier score. These results are exploratory comparisons on one
series, not evidence of generalization to independent matches.

All 555 re-exported prediction rows matched their original provenance and
probabilities exactly (maximum absolute probability difference: 0). Browser
checks passed for the 20-second boundary, five-second updates, backward seeking,
model switching, training/validation labels, both maps, and retention of the last
available prediction after round end. No browser errors or warnings were captured.

The local preview is [127.0.0.1:8766](http://127.0.0.1:8766/). It was left running
for inspection; the original 8765 viewer was not changed. It serves saved JSON,
not a running ML model. The environment and outputs are retained in the ignored
`runs/ml-validation-20261029/` directory (approximately 753 MiB at completion).
The local [validation record](../runs/ml-validation-20261029/validation.json)
contains exact metrics, file hashes, and verification details. Installation
reports and the run manifest are alongside it. Those ignored files are not
included in Git. Removal instructions are below; cleanup has not been performed.

## Dependencies

The proposed target is the existing Homebrew CPython 3.12 interpreter on this Mac
(arm64). These wheels require macOS 14 or later. The machine reports macOS 27.0.
This is a platform-specific candidate set, not a tested cross-platform lockfile.

| Direct dependency | Proposed version | Purpose |
| --- | --- | --- |
| [NumPy](https://pypi.org/project/numpy/2.5.4/) | 2.5.4 | Arrays and baseline inputs |
| [scikit-learn](https://pypi.org/project/scikit-learn/1.9.1/) | 1.9.1 | Weighted scalers and logistic regression |
| [PyTorch](https://pypi.org/project/torch/2.15.0/) | 2.15.0 | Temporal CNN, optimization, and inference |
| [safetensors](https://pypi.org/project/safetensors/0.8.0/) | 0.8.0 | Local tensor checkpoint storage |

Runtime dependencies are also pinned: SciPy 1.18.1, joblib 1.6.0,
narwhals 2.28.0, threadpoolctl 3.7.0, filelock 4.2.4,
typing-extensions 4.16.0, setuptools 84.2.1, SymPy 1.14.0,
NetworkX 3.7, Jinja2 3.1.6, fsspec 2026.9.0, MarkupSafe 3.0.4,
mpmath 1.3.0, and cloudpickle 3.1.2. SymPy requires mpmath below 1.4;
using the newest mpmath release would violate that requirement.

[The installation file](../requirements-validation-macos-arm64-py312.txt) names
all 18 wheel URLs and SHA-256 hashes. [The metadata record](dependency-review.json)
contains the version-specific PyPI sources, runtime requirements, wheel sizes,
and hashes. The pre-approval review fetched only public metadata; the approved
execution subsequently downloaded and installed the pinned wheels. They total 184,541,463 bytes
(about 185 MB compressed); installed files and temporary space require more.
CUDA/Linux packages, optional extras, notebooks, plotting libraries, and remote
model clients are not part of this plan. Bundled native libraries inside wheels
are still third-party software.

The installer is also pinned: **pip 26.2.2**, a 1,819,850-byte wheel. The existing
Python bundles pip 26.1, which has two published security issues. The commands
below bypass it. See the [security review](security-review.md) and
[machine-readable evidence](security-review.json). Total proposed download size,
including pip, is 186,361,313 bytes.

## Approved commands used for this run

Run commands in order and stop if any fails. Do not substitute versions, allow
source builds, add dependencies, or overwrite an existing run to work around a
failure. This is a three-epoch pipeline check, not a tuned experiment. The setup,
installation, test, training, export, and preview commands below were executed.
The directory now exists, so rerunning setup will intentionally stop. The optional
removal command was not executed.

Start a separate shell so cache and thread settings disappear when it exits.
Use one new, Git-ignored directory for the environment and generated files:

```bash
/bin/zsh -f
set -e
umask 077
cd /Users/arigladstone/Documents/Valorant-Round-Outcome-Modeling
export VALORANT_VALIDATION_ROOT="$PWD/runs/ml-validation-20261029"
test ! -L runs
mkdir -p runs
mkdir "$VALORANT_VALIDATION_ROOT"
mkdir "$VALORANT_VALIDATION_ROOT/tmp" "$VALORANT_VALIDATION_ROOT/cache"
export TMPDIR="$VALORANT_VALIDATION_ROOT/tmp"
export XDG_CACHE_HOME="$VALORANT_VALIDATION_ROOT/cache"
export TORCH_HOME="$VALORANT_VALIDATION_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$VALORANT_VALIDATION_ROOT/cache/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$VALORANT_VALIDATION_ROOT/cache/torchinductor"
export JOBLIB_TEMP_FOLDER="$VALORANT_VALIDATION_ROOT/tmp"
export PIP_CONFIG_FILE=/dev/null
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

Download only the pinned installer wheel with the system standard library.
This rejects redirects, unexpected size, and hash mismatches before writing the
wheel. No downloaded code is executed in this step. It also creates the local,
hashed bootstrap requirement; the installer will not need an index to install itself.

```bash
/usr/bin/python3 -I -S -B - <<'PY'
import hashlib
import os
from pathlib import Path
import urllib.error
import urllib.request

root = Path(os.environ['VALORANT_VALIDATION_ROOT'])
url = 'https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl'
expected = 'fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70'
size = 1819850

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise urllib.error.HTTPError(request.full_url, code, 'Redirect refused', headers, fp)

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
with opener.open(url, timeout=60) as response:
    payload = response.read(size + 1)
if len(payload) != size or hashlib.sha256(payload).hexdigest() != expected:
    raise SystemExit('Installer size/hash mismatch; stopped.')
wheel = root / 'pip-26.2.2-py3-none-any.whl'
with wheel.open('xb') as output:
    output.write(payload)
with (root / 'bootstrap-requirements.txt').open('x') as output:
    output.write(f'pip @ {wheel.as_uri()} --hash=sha256:{expected}\n')
print('Pinned installer downloaded and hash verified; no package code executed yet.')
PY
```

Create an environment **without** the bundled pip. Then execute only the verified
pip wheel to install itself locally. The bootstrap rechecks its hash immediately
before importing it. This is the first execution of newly downloaded third-party
code in the plan. No global Python/pip upgrade is performed.

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -I -S -B -m venv \
  --without-pip "$VALORANT_VALIDATION_ROOT/venv"
"$VALORANT_VALIDATION_ROOT/venv/bin/python" -I -B - <<'PY'
import hashlib
import os
from pathlib import Path
import runpy
import sys

root = Path(os.environ['VALORANT_VALIDATION_ROOT'])
wheel = root / 'pip-26.2.2-py3-none-any.whl'
expected = 'fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70'
if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected:
    raise SystemExit('Installer hash mismatch; stopped.')
sys.path.insert(0, str(wheel))
sys.argv = ['pip', '--isolated', '--disable-pip-version-check', 'install',
            '--no-index', '--no-deps', '--only-binary=:all:', '--require-hashes',
            '--no-cache-dir', '--no-input', '--no-compile',
            '--report', str(root / 'bootstrap-install.json'),
            '-r', str(root / 'bootstrap-requirements.txt')]
runpy.run_module('pip', run_name='__main__', alter_sys=True)
PY
```

Install only the 18 explicit, hashed wheel URLs. `--no-deps` prevents additional
packages being selected; `--no-index` does not prevent downloading these explicit
URLs. Configuration files are disabled for this shell, and pip uses isolated mode.
`pip check` must pass before proceeding. See pip's
[secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/) and
[configuration](https://pip.pypa.io/en/stable/topics/configuration/) documentation.

```bash
"$VALORANT_VALIDATION_ROOT/venv/bin/python" -I -B -m pip \
  --isolated --disable-pip-version-check install \
  --no-index --no-deps --only-binary=:all: --require-hashes \
  --no-cache-dir --no-input --no-compile \
  --report "$VALORANT_VALIDATION_ROOT/dependency-install.json" \
  -r requirements-validation-macos-arm64-py312.txt
"$VALORANT_VALIDATION_ROOT/venv/bin/python" -I -B -m pip \
  --isolated --disable-pip-version-check check
"$VALORANT_VALIDATION_ROOT/venv/bin/python" -I -B -m pip \
  --isolated --disable-pip-version-check list --format=json \
  > "$VALORANT_VALIDATION_ROOT/dependencies-installed.json"
```

Run the standard-library suite on the new interpreter, then the two rolling ML
tests. The latter execute the libraries, fit synthetic-data baselines, perform
a CNN backward pass, and save/load a temporary local safetensors checkpoint:

```bash
"$VALORANT_VALIDATION_ROOT/venv/bin/python" -E -S -B -m unittest discover -s tests -v
"$VALORANT_VALIDATION_ROOT/venv/bin/python" -E -s -B -m unittest discover \
  -s tests_ml -p test_rolling_models.py -v
```

Fit the three baselines and run the CNN for three epochs. The seed and map split
are fixed. Training uses CPU tensors, one PyTorch thread, and no data-loader
workers. Each round contributes total sample weight one. The best CNN checkpoint
is selected by validation log loss averaged by round, then reloaded locally:

```bash
"$VALORANT_VALIDATION_ROOT/venv/bin/python" -E -s -B train_rolling.py \
  --data-dir data --output-dir "$VALORANT_VALIDATION_ROOT/rolling-smoke-5s" \
  --smoke-test --split-seed 42 --model-seed 42 --include-cnn --epochs 3
```

Exercise the export CLI by loading only artifacts from that newly created run:

```bash
"$VALORANT_VALIDATION_ROOT/venv/bin/python" -E -s -B predict_rolling.py \
  "$VALORANT_VALIDATION_ROOT/rolling-smoke-5s" data \
  > "$VALORANT_VALIDATION_ROOT/rolling-smoke-5s/reexport.jsonl"
```

Compare the exports with the system standard library. Provenance must match
exactly and probabilities must agree within 0.000001. A mismatch stops the plan:

```bash
/usr/bin/python3 -I -S -B - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ['VALORANT_VALIDATION_ROOT']) / 'rolling-smoke-5s'
a = [json.loads(line) for line in (root / 'predictions.jsonl').read_text().splitlines()]
b = [json.loads(line) for line in (root / 'reexport.jsonl').read_text().splitlines()]
assert len(a) == len(b) and len(a) > 0
for original, exported in zip(a, b):
    p, q = original.pop('probabilities'), exported.pop('probabilities')
    assert original == exported and p.keys() == q.keys()
    assert all(abs(p[name] - q[name]) <= 1e-6 for name in p)
print(f'Checked {len(a)} prediction rows and their provenance.')
PY
```

Display predictions in a separate loopback preview, leaving the current 8765
viewer available. This server reads JSON and never opens models:

```bash
/usr/bin/python3 -E -S -B serve.py --data-dir data --port 8766 \
  --rolling-predictions "$VALORANT_VALIDATION_ROOT/rolling-smoke-5s/predictions.jsonl" \
  --rolling-run-manifest "$VALORANT_VALIDATION_ROOT/rolling-smoke-5s/run.json"
```

Open [the prediction preview](http://127.0.0.1:8766/) only after the server starts.
Check 19.95s, 20s, 25s, backward seeking, model switching, partition labels, and
the latest available prediction after a round ends. The visualizer must not show
a prediction before its observation time. Stop this temporary server with Ctrl+C
when finished. If keeping the outputs for later viewing, exit the separate shell
without performing the optional cleanup below.

## Removal commands, when the outputs are no longer wanted

First stop the temporary preview and any validation process. This removes only
the dedicated validation directory, including its environment, caches, downloaded
installer, fitted models, and predictions. It does not remove the entire `runs/`
directory or uninstall global packages. The path checks refuse symlink roots.
Run from the repository directory; this command is also only proposed, not executed:

```bash
/usr/bin/python3 -I -S -B - <<'PY'
from pathlib import Path
import shutil
repo = Path('/Users/arigladstone/Documents/Valorant-Round-Outcome-Modeling')
root = repo / 'runs' / 'ml-validation-20261029'
if Path.cwd().resolve() != repo.resolve():
    raise SystemExit('Run from the reviewed repository directory.')
if root.is_symlink() or root.parent.is_symlink() or not root.is_dir():
    raise SystemExit('Expected an existing, nonsymlink validation directory.')
shutil.rmtree(root)
assert not root.exists()
print('Removed the dedicated validation directory.')
PY
```

Exit the separate shell to discard its temporary environment settings. The
empty `runs/` parent, if created by this plan, can remain. Cleanup also means the
8766 prediction preview cannot be restarted using those deleted files.

## Security risks and limits

- Downloads go to `files.pythonhosted.org`, which receives the request URLs and
  normal network metadata. The plan has no dataset-upload step. It is not a
  network sandbox; dependency code retains the account's normal network access.
- Python/native dependency code executes during validation. A venv separates
  Python packages, but does not restrict filesystem access or provide OS isolation.
  See [Python venv](https://docs.python.org/3.12/library/venv.html).
- Binary-only installation avoids source builds. Hash checks bind downloads to
  the reviewed metadata; neither hashes nor clean advisory results prove code is
  safe. OSV returned no matching advisories for the 18 pins or pip 26.2.2 on the
  review date. PyTorch 2.15.0 is a recent release and is untested here. Archive
  contents, vendored native libraries, and all package source have not been audited.
- joblib/cloudpickle are transitive dependencies with serialization functionality.
  The project does not use them to load saved models. JSON and safetensors avoid
  pickle model loading but do not eliminate parser/native-library vulnerabilities.
- The small CPU run consumes RAM/disk and may heat the machine. No GPU driver
  setup, `sudo`, global Python changes, remote checkpoints, API keys, cloud
  training, or public hosting are part of the commands.
- Expected environment, run, cache, and temporary files are confined by path
  choices to the dedicated directory. These settings are conventions, not access
  controls. **Deleting it is not guaranteed to erase all traces or undo malicious
  behavior.** OS logs, crash reports, swap, backups, browser/shell history, and
  remote download logs can remain. See the [removal assessment](security-review.md#removal).

Any code change, hash mismatch, dependency conflict, or unexpected behavior stops
execution for review. The metrics above describe only the completed limited run.
The included data has one series: `--smoke-test` holds out one map and creates no
test partition. Its scores check pipeline behavior and offer exploratory
comparisons; they do not establish generalization to independent matches.
Subsequent longer training, repeated seeds and feature ablations are documented
in the [v2 execution record](controlled-experiment.md). Additional independent
data and a held-out test remain future work.
