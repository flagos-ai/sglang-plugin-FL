# Hygon derived SGLang wheel

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
- RECORD paths may contain literal commas (triton config JSONs like
  `lora_expand,K=1024,...json`); parsing splits from the right.

## Publication policy

The derived wheel uses a NEW local version (`0.5.18+flagos.hcu1`). Never
overwrite or republish `0.5.18+flagos`, never change only the filename,
and never let one name/version pair carry different content. Where the
derived wheel is published (the Hygon platform index) is an owner decision
for the design review; until then `stage_wheels.sh` derives it locally in
the disposable build context and the wheel is never committed.

Generated wheels, unpack trees, manifests and provenance JSONs are build
evidence: keep them under `$TMPDIR`/CI artifacts, out of the repository.
