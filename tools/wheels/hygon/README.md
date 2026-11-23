# Hygon derived platform wheels

This directory owns the artifact tooling that turns the published FlagOS
SGLang wheel plus the reviewed HCU gating patch into a NEW derived wheel,
so the product image installs a normal wheel and never patches
site-packages at build time.

```
sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl   (input, SHA256-pinned)
sglang-v0.5.18-hcu.patch                             (input, reviewed)
        |
        |  derive_sglang_wheel.py
        v
sglang-0.5.18+flagos.hcu1-cp310-cp310-linux_x86_64.whl  (+ provenance JSON)

xgrammar-0.2.1-cp310-...-manylinux_2_28_x86_64.whl  (input, SHA256-pinned)
        |
        |  derive_metadata_wheel.py   (drops the active HCU triton requirement only)
        v
xgrammar-0.2.1+flagos.hcu1-cp310-...-manylinux_2_28_x86_64.whl  (+ provenance JSON)
```

## Files

- `sglang-v0.5.18-hcu.patch` — the reviewed HCU-vs-ROCm gating patch. It
  moved here from `docker/hygon/` because it is an INPUT to the derived
  wheel, not an image-assembly step. The empty-image source-build line
  (`docker/hygon/empty-0.5.18.containerfile`) consumes the same file from
  this location; the two-wheel runtime containerfile does not reference it
  at all.
- `derive_sglang_wheel.py` — derives the wheel (contract below).
- `verify_derived_sglang_wheel.py` — standalone strict diff gates; also
  imported by the derive script as its post-build self-check.
- `test_derive_sglang_wheel.py` — pytest: synthetic wheels cover the
  error branches; the success paths run the SHA-pinned real wheel.
- `derive_metadata_wheel.py` / `verify_metadata_wheel.py` /
  `test_derive_metadata_wheel.py` — the metadata-only correction and its
  independent verifier (last section).
- `publication_manifest.py` — writes the publication manifest of a staging
  context (see "Content pins, pre-derived wheels, publication").

## Derive contract

```
python3 tools/wheels/hygon/derive_sglang_wheel.py \
  --input-wheel <pinned wheel> \
  --expected-input-sha256 1fe8a560353fd15aea7194869f1fd50e1fdbedd8fc1074db52b9a13028756800 \
  --patch tools/wheels/hygon/sglang-v0.5.18-hcu.patch \
  --output-dir <dir> \
  --derived-version 0.5.18+flagos.hcu1 \
  --source-commit <HEAD>
```

1. the input wheel must exist and hash exactly to the pinned SHA256;
2. its distribution must be `sglang`, version exactly `0.5.18+flagos`
   (checked in METADATA and in the filename);
3. the derived version must differ and be exactly `0.5.18+flagos.hcu1`;
4. the wheel is unpacked to a mktemp directory outside the repository;
5. the patch is `git apply --check`-ed then applied there (`-p2`);
6. only these changes are made:
   - the METADATA `Version` line,
   - the `*.dist-info` directory name (`sglang-0.5.18+flagos.hcu1.dist-info`),
   - the wheel filename,
   - RECORD (rebuilt; see the quirk note below),
   - `sglang/_version.py` ONLY when the original wheel's copy carried the
     input distribution version — the published wheel's copy hardcodes
     `0.5.18` (no local segment), so it stays byte-identical and
     `sglang.__version__` remains `'0.5.18'`, exactly as in the original;
7. the output zip is deterministic: input member order, per-member
   timestamps/modes/attributes are preserved, everything is DEFLATE
   level 9, RECORD is written in sorted order — two independent
   derivations are byte-identical (proven by the tests);
8. the finished wheel is re-verified through the strict diff gates before
   the script reports success; a failed gate deletes the output;
9. a provenance JSON (input/patch/derived SHA256s, patch file list,
   source commit) is written next to the output wheel — disposable
   evidence, never committed;
10. intermediate directories are removed on exit.

## Strict diff gates (verify_derived_sglang_wheel.py)

Allowed to differ between the pinned input wheel and the derived wheel:

- the SGLang Python files the patch actually modifies (parsed from the
  patch's `+++ b/python/sglang/...` side; do not assume a count);
- the METADATA `Version` line (raw-byte check: every other byte identical);
- the `.dist-info` directory name;
- RECORD;
- `sglang/_version.py` under the policy above.

Everything else must be byte-identical, including all native payload
(`.so`/`.a`/`.h`/...), every unpatched Python file, `Requires-Dist`,
`Provides-Extra`, entry points, `WHEEL`, `top_level.txt`, licenses, and
the wheel tags. Additional structural proofs: the derived RECORD's
hash/size is valid for every member; filename / METADATA / dist-info
directory / package version are mutually consistent; the forward patch no
longer applies to the derived wheel; the reverse patch passes `--check`;
and a fresh unpack of the original wheel patched with the input patch
reproduces the derived wheel's patched files byte-for-byte.

## Input-wheel quirks (observed, handled)

- The published wheel's RECORD references a stale
  `sglang-0.5.18.dist-info/` prefix although its archive ships the
  `sglang-0.5.18+flagos.dist-info` directory. The derived RECORD is
  rebuilt against the real tree, so it names the actual new dist-info
  directory.
- RECORD is comma-separated CSV: the wheel ships 536 members whose paths
  contain literal commas (triton config JSONs like
  `lora_expand,K=1024,...json`). The published wheel writes them
  UNQUOTED (pip tolerates it with per-line warnings); the derived RECORD
  quotes them via ``csv.writer`` so every row parses back through
  ``csv.reader`` to exactly three fields. The verifier REJECTS unquoted
  comma rows (regression-tested).
- 6 dist-info members are STORED in the input wheel; the deterministic
  repack preserves each member's compression kind (STORED stays STORED,
  DEFLATED members are recompressed at level 9).

## Provenance and image binding

The provenance JSON written next to the wheel carries, besides the
input/patch/derived SHA256s, a per-file manifest of every patched file
(hashes read back from the FINISHED wheel). `stage_wheels.sh` ships the
manifest with the build context and `build_two_wheel.sh` hands both the
expected wheel digest and the manifest to `verify_image.sh`, which
asserts the `org.flagos.ci.sglang-wheel-sha256` OCI label equals the
digest and that the INSTALLED patched files match the manifest
byte-for-byte - a same-version, different-content wheel cannot pass.

## Publication policy

The derived wheel uses a NEW local version (`0.5.18+flagos.hcu1`). Never
overwrite or republish `0.5.18+flagos`, never change only the filename,
and never let one name/version pair carry different content: once `hcu1`
is published, any content change requires `hcu2` - the version never gets
reused.

Long-term ownership direction for the design review: the derive tooling
stays in this repository as the artifact PRODUCTION tool, but release
images should consume an already-published, immutable, SHA-pinned
derived wheel from the artifact service (same pinning discipline as the
input wheel) instead of deriving during image assembly. Staging-time
derivation is the development-candidate mode used until that publication
exists.

Generated wheels, unpack trees, manifests and provenance JSONs are build
evidence: keep them under `$TMPDIR`/CI artifacts, out of the repository.

## Content pins, pre-derived wheels, publication

One version carries exactly one content. `docker/hygon/stage_wheels.sh`
enforces it at staging time with the digests the two derived wheels hash
to from this checkout's tools and the pinned inputs:

| wheel | SHA256 |
|---|---|
| `sglang-0.5.18+flagos.hcu1-cp310-cp310-linux_x86_64.whl` | `edcae8c0678290ec68b83d8ec38537d0ef7852b3f8f183984126bba9e8639b5f` |
| `xgrammar-0.2.1+flagos.hcu1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` | `c2deb2853ac98cbaf63ecfa7f55dd00c54e25c9c43a34e9edb234f84e316d1a2` |

- a derive that produces anything else fails staging: different content
  needs the next local version, never a re-published `hcu1`. The pin covers
  the compressed bytes too, so a tooling or deflate-implementation change
  that alters the archive stops the build instead of silently shipping a
  second content under the same version;
- an already-derived wheel (a published artifact, or a secured local copy)
  can be staged instead of deriving: `SGLANG_DERIVED_WHEEL=<path>` /
  `XGRAMMAR_DERIVED_WHEEL=<path>`. Staging copies it together with its
  sibling `.provenance.json` and verifies it against the same pin;
  nothing is unpacked, patched or repacked inside an image;
- `publication_manifest.py --context <staging-context-dir>` writes
  `publication-manifest.json`: per artifact the distribution, version,
  filename, size, SHA256 and the suggested Nexus location
  (`flagos-pypi-hygon`, `packages/<name>/<version>/<file>` - the layout
  the published input wheel already uses). Every entry is marked LOCAL
  DEVELOPMENT CANDIDATE until the artifact owner publishes it;
- a release build consumes the PUBLISHED wheels by pinning these same
  SHA256 values. The post-build acceptance requires lineage, not equality:
  the commit recorded in a wheel's provenance must exist in the repository
  and be an ancestor of the image's source commit
  (`docker/hygon/scripts/provenance_lineage.py`), so a wheel derived - or
  published - before the image passes while one from an unrelated history
  does not.

## Metadata-only dependency correction

`derive_metadata_wheel.py` and `verify_metadata_wheel.py` implement the same
immutable-artifact policy for third-party metadata defects without changing
their payload. The current HCU use is xgrammar 0.2.1: its Linux/x86_64 wheel
declares a standalone `triton` distribution even though the Hygon Runtime owns
the compiler capability as `/opt/flagtree` and deliberately has no Triton
distribution metadata.

The staging flow SHA-pins the official cp310 x86_64 wheel, removes exactly:

```text
Requires-Dist: triton; platform_system == "Linux" and platform_machine == "x86_64"
```

and emits `xgrammar==0.2.1+flagos.hcu1`. The independent verifier requires the
line to be active for the target environment and present exactly once; permits
only that deletion, the Version/dist-info rename and rebuilt RECORD; and proves
all payload/native files, tags, ZIP member order and per-member compression are
unchanged. The provenance manifest binds every wheel member to the candidate
image through `org.flagos.ci.xgrammar-wheel-sha256`; post-build acceptance
compares every installed member except installer-owned RECORD byte-for-byte.

This is staging-time development-candidate production, not license to rewrite
arbitrary downloaded wheels during image assembly. The release endpoint is an
immutable corrected wheel in the artifact service. A content change requires a
new local version (`hcu2`, and so on); `hcu1` must never be overwritten.
