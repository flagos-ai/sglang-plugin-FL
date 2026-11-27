# Hygon derived platform wheels

Artifact tooling that turns two published wheels into the HCU platform wheels
the two-wheel image installs, so the image installs normal wheels and never
patches site-packages at build time. `docker/hygon/stage_wheels.sh` runs both
derivations and checks the results against pinned SHA256 values.

```
sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl  (input, SHA256-pinned)
sglang-v0.5.18-hcu.patch                           (input, reviewed)
        |  derive_sglang_wheel.py
        v
sglang-0.5.18+flagos.hcu1-cp310-cp310-linux_x86_64.whl

xgrammar-0.2.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
        |  derive_metadata_wheel.py
        v
xgrammar-0.2.1+flagos.hcu1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

## Files

- `sglang-v0.5.18-hcu.patch` - the reviewed HCU-vs-ROCm gating patch. The
  empty-image source-build line (`docker/hygon/empty-0.5.18.containerfile`)
  consumes the same file.
- `derive_sglang_wheel.py`, `verify_derived_sglang_wheel.py` - the SGLang
  derivation and its strict diff gates (also the derive script's self-check).
- `derive_metadata_wheel.py`, `verify_metadata_wheel.py` - the xgrammar
  metadata correction and its independent verifier.

## SGLang derivation contract

```
python3 tools/wheels/hygon/derive_sglang_wheel.py \
  --input-wheel <pinned wheel> \
  --expected-input-sha256 1fe8a560353fd15aea7194869f1fd50e1fdbedd8fc1074db52b9a13028756800 \
  --patch tools/wheels/hygon/sglang-v0.5.18-hcu.patch \
  --output-dir <dir> --derived-version 0.5.18+flagos.hcu1 --source-commit <HEAD>
```

1. The input wheel must hash to the pinned SHA256 and be `sglang==0.5.18+flagos`
   (METADATA and filename); the derived version must be `0.5.18+flagos.hcu1`.
2. The wheel is unpacked to a mktemp directory outside the repository and the
   patch is `git apply --check`-ed, then applied there (`-p2`).
3. Only these change: the METADATA `Version` line, the `*.dist-info` directory
   name, the wheel filename, RECORD, and `sglang/_version.py` only when the
   original carried the input version (the published wheel hardcodes `0.5.18`,
   so it stays byte-identical and `sglang.__version__` stays `'0.5.18'`).
4. The output zip is deterministic: input member order and per-member
   timestamps, modes, attributes and compression kind are preserved, DEFLATE
   members are written at level 9, RECORD is sorted. Two derivations of the
   same inputs are byte-identical.
5. The finished wheel is re-verified through the strict diff gates before the
   script reports success; a failed gate deletes the output.

Allowed to differ between the input and the derived wheel: the SGLang files
the patch modifies (parsed from the patch's `+++` side), the METADATA `Version`
line (every other byte identical), the `.dist-info` directory name, RECORD and
`sglang/_version.py` as above. Everything else must be byte-identical,
including all native payload, unpatched Python files, `Requires-Dist`,
`Provides-Extra`, entry points and the wheel tags. The verifier also proves
that the derived RECORD's hash and size are valid for every member, that the
forward patch no longer applies, that the reverse patch passes `--check`, and
that a fresh unpack of the original patched with the same patch reproduces the
patched files byte for byte.

Input-wheel quirks the derivation handles:

- The published RECORD references a stale `sglang-0.5.18.dist-info/` prefix
  although the archive ships `sglang-0.5.18+flagos.dist-info`; the derived
  RECORD is rebuilt against the real tree.
- RECORD is comma-separated CSV and 536 member paths contain literal commas
  (triton config JSONs). The published wheel leaves them unquoted; the derived
  RECORD quotes them through `csv.writer`, and the verifier rejects unquoted
  comma rows.
- 6 dist-info members are STORED in the input wheel; the repack keeps each
  member's compression kind.

## xgrammar metadata correction

xgrammar 0.2.1 declares a standalone `triton` distribution on Linux/x86_64,
but the Hygon Runtime owns the compiler as the `/opt/flagtree` module tree and
ships no triton distribution metadata. `derive_metadata_wheel.py` removes
exactly

```
Requires-Dist: triton; platform_system == "Linux" and platform_machine == "x86_64"
```

and emits `xgrammar==0.2.1+flagos.hcu1`. Its verifier requires the line to be
present exactly once and active for the target environment, permits only that
deletion, the Version/dist-info rename and the rebuilt RECORD, and proves that
every payload and native file, the tags, the ZIP member order and the
per-member compression are unchanged.

## One version, one content

A derived version is bound to one content. `docker/hygon/stage_wheels.sh` pins:

| wheel | SHA256 |
|---|---|
| `sglang-0.5.18+flagos.hcu1-cp310-cp310-linux_x86_64.whl` | `edcae8c0678290ec68b83d8ec38537d0ef7852b3f8f183984126bba9e8639b5f` |
| `xgrammar-0.2.1+flagos.hcu1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` | `c2deb2853ac98cbaf63ecfa7f55dd00c54e25c9c43a34e9edb234f84e316d1a2` |

A derivation that produces anything else fails staging: different content
needs the next local version (`hcu2`), and `hcu1` is never republished with
other content. Never overwrite `0.5.18+flagos`. Generated wheels and unpack
trees are build evidence: keep them under `$TMPDIR`, out of the repository.

Staging-time derivation is the development-candidate mode. A release image
consumes the published, immutable, SHA-pinned derived wheels from
`resource.flagos.net` instead (`SGLANG_DERIVED_WHEEL` / `XGRAMMAR_DERIVED_WHEEL`
in `stage_wheels.sh`).
