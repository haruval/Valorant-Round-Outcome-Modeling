# ML validation security review

Pre-execution review dated October 29, 2026. The owner subsequently approved this
plan, and the limited run passed. See [execution results](ml-validation.md#completed-run).
The findings below describe the review before package downloads and execution.

**Recommendation:** the revised, limited CPU validation plan is reasonable to run
with the owner's approval. It has the ordinary risks of installing and executing
third-party Python/native packages. This review does not establish that the code
is harmless or that every effect on the machine can be reversed.

## Findings

The original environment setup would have used pip 26.1 bundled with the existing
Homebrew Python 3.12.13 installation. OSV reports two distinct vulnerabilities for
that version (four records, including aliases):

| Advisory | Issue | Fixed version |
| --- | --- | --- |
| [CVE-2026-13346 / GHSA-qwm4-qh6w-59xr](https://osv.dev/vulnerability/GHSA-qwm4-qh6w-59xr) | A malicious package index can use double-encoded URLs to cause writes outside the intended location. | 26.2.0 |
| [CVE-2026-8643 / GHSA-wf93-45jw-7689](https://osv.dev/vulnerability/GHSA-wf93-45jw-7689) | Crafted wheel entry-point names can cause scripts to be written outside the installation directory. | 26.1.2 |

The upstream fixes are [URL decoding](https://github.com/pypa/pip/pull/14110) and
[entry-point path validation](https://github.com/pypa/pip/pull/14000). Direct wheel
URLs and `--no-index` limit exposure to the first issue, but using an affected
installer is unnecessary. The revised commands create a venv **without pip**,
verify the SHA-256 of pip 26.2.2 before executing it, and install that patched pip
from its local wheel. They do not modify Homebrew or the global Python installation.

For the 18 pinned ML/runtime packages:

- OSV returned no matching advisories for the exact package/version queries.
  The replacement pip 26.2.2 query also returned none. This is an advisory-database
  check, not a malware scan or proof that no vulnerabilities exist.
- Each selected wheel's URL, SHA-256, size, and non-yanked status matched the
  current PyPI metadata. Dependency constraints were checked from metadata;
  installed compatibility still requires `pip check` and the ML tests.
- PyPI supplied provenance records for 13 wheels. Their decoded subject hashes
  matched the pinned metadata. **Signatures and trust chains were not verified.**
  No provenance record was returned for torch, safetensors, setuptools, fsspec,
  or mpmath. Missing provenance alone is not evidence of malicious code.
- No wheel was downloaded during this review. The actual archive contents,
  embedded native libraries, and complete dependency source code were not audited.
  PyTorch 2.15.0 was published October 28, 2026 and remains untested here.

The exact query results, artifact metadata, reported publishers, and hashes of the
reviewed project files are in [security-review.json](security-review.json).
See PyPI's [integrity API](https://docs.pypi.org/api/integrity/) and
[attestation security model](https://docs.pypi.org/attestations/security-model/)
for the distinction between provenance and code safety.

## Project execution paths

I read the rolling training, inference, baseline, CNN, ML test, and preview-server
paths. The reviewed project code contains no package/model downloads, shell
commands, or data-upload step in those paths. It uses the local event file and
only newly created local artifacts. Baselines are validated JSON; CNN weights use
safetensors with size/schema checks. There is no `pickle.load`, `joblib.load`, or
`torch.load` in this path. Artifact hashes are checked before re-export.

The proposed test fits synthetic baselines and exercises a CNN backward pass and
checkpoint round trip. The proposed real-data run fits three baselines and trains
the small CNN for three epochs on CPU. The output directory must be new. It uses
one PyTorch training thread and no data-loader workers. The visualizer serves
JSON on `127.0.0.1:8766`; it does not load models.

These observations describe the project code, not every behavior of dependencies.
Native parsers can have bugs, and malicious dependency code could access files or
the network with the account's permissions. A venv separates Python installations;
it is not an OS sandbox. [Python venv documentation](https://docs.python.org/3.12/library/venv.html).
The current OS, existing interpreter, and all globally installed software were
not audited. There is no proposed `sudo`, driver installation, service registration,
shell startup-file change, or global package installation.

## Removal

The revised plan puts the environment, installer wheel, installation records,
test temporary files, expected caches, fitted artifacts, and predictions under
`runs/ml-validation-20261029/`, which is ignored by Git. It disables pip's cache
and redirects temporary/cache locations for the validation shell. After stopping
the preview and validation processes, that directory can be deleted without
uninstalling anything from the system interpreter. The existing data, code, and
8765 preview are independent of it. Removing it also removes its predictions.

**This is removal of the planned files, not a guarantee of erasing every trace.**
Cache environment variables are conventions, not access controls. OS logs,
crash reports, swap, browser/shell history, backups, and remote download logs may
remain. A compromised package could write outside the directory; deleting a venv
would not necessarily undo that. No global cache purge or system cleanup is proposed.

If guaranteed separation from host files is a requirement, use a disposable VM
without host directory mounts, with only a copy of the dataset and network access
disabled after package acquisition. That is a separate setup proposal; none was
created here.

## Approval scope

The [revised exact commands](ml-validation.md) cover the patched installer,
18 pinned runtime wheels, tests, three training epochs, local artifact reload,
prediction comparison, and loopback preview. No installation or ML execution had
occurred at the time of this review. Any hash mismatch, missing wheel, dependency conflict, test failure, or
unexpected behavior stops the plan for review; it does not authorize substituting
packages or extending the experiment.
