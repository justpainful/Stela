# Performance log

Running log of the cloud agent's sessions. Numbers say which machine they
come from. Owner-boot numbers come from `stela log D:`.

## 2026-11-21, session 1 (cloud)

Machine: cloud container, 4 vCPU Intel Xeon (AVX-512, AVX-512 FP16, AMX),
15 GB. Not the owner's i7-12700KF; no timing here is the owner's.

### Base

`performance/daily-driver` at `0bc7857`. Upstream `tylertms/Stela` main was
32 commits ahead (`140ffaf`).

### Baseline tests (daily-driver `0bc7857`)

- `stela-compiler` lib: 1583 passed, 7 failed, 21 ignored.
  All 7 failures are AVX-512 FP16 (EVEX maps 5 and 6) cases that only run
  on a host that has FP16:
  `modeled::tests::models_match_the_host_instructions` (map 6, opcode 0x62,
  sample 187: two result bytes and the flags differ), 
  `simd::arithmetic::half_tests::half_arithmetic_lanes_match_the_reference`,
  and five `simd::half_integer_tests::half_to_integer_*` ("covered input
  exited"). The owner's CPU has no AVX-512, so these do not reach the
  owner's machine, but they are real model gaps on FP16 hosts.
- The first run stopped there (cargo stops at the first failing crate);
  the other suites are run with `--no-fail-fast` below.
- UEFI clippy of `stela-boot` failed on daily-driver: unused
  `Assembler::mfence`. Fixed by cherry-picking `agent/atomics` `fda312d`.

### Changes

Branch `claude/magical-lamport-hekj7b` (to be merged into
`performance/daily-driver`):

1. `5329e69` perf(nvme): a host NVMe command moves up to 1 MiB within MDTS
   (was 128 KiB); the bounce pool is 2056 pages (8 MiB, eight largest
   commands; was 256 pages, 1 MiB). A 1 MiB guest ANS transfer is one host
   command instead of eight where MDTS >= 8. Command contents, FUA, flush,
   discard, ordering and block ranges are unchanged.
   Tests: `stela-driver-storage` lib 314 passed.
2. `183fcc2` fix(storage): the "no completion interrupt, polling" verdict
   is no longer final. A disk found silent is watched in windows of 32
   completions; 8 or more that each follow a signal make it heard again.
   Session 4 judged the NVMe silent at about 36 s, before the guest ran,
   and polled it for the whole boot.
   Tests: `stela-boot-core` lib 1128 passed (2 new).
3. cherry-pick of `fda312d`: removes the unused MFENCE helper, so the
   UEFI clippy lane passes again.

Clippy: UEFI `stela-boot`, `stela-driver-storage`, `stela-boot-core
--features native-acpi`, and host `--all-targets` for both crates: clean.

Branch `agent/merge-upstream-1121` (`a5d87ab`): upstream main merged into
daily-driver. Conflicts:
- `translations/writer.rs`: upstream reads packs on demand and removed the
  writer's arena, so the arena compaction on `agent/pack-writer`
  (`83f9307`, `d2c75b2`, `1df4b2e`) is obsolete (upstream wins). The
  per-part step timing (`a83b223`) is ported onto upstream's step:
  `parts_us=` now names govern, begin, hold, delta, merge, plan, fetch,
  write, commit.
- `system/memory.rs`: upstream's `sizes_with_room` with the branch's
  runner count.
- engine `tests.rs`: upstream's `real_counter` replaces the branch's
  `stamped`; the run loop bench uses it.

### Needs an owner boot

- Lines: `HOST DISK serial=... completion interrupt heard again` (new),
  `HOST DISK serial=... no completion interrupt, polling`,
  `HOST DISK STAGED ... calls/total_us start=... owed=...`,
  `TRANSLATIONS writer ... parts_us=...`, `PES STALLS`, `AUDIO out=`.
- The latest boot's log (stick build `eeeb70f3f`) is still needed as the
  M1 baseline for audio stutter and freezes.

### Next

- M1 audio: read the `AUDIO out=` fields of the `eeeb70f3f` boot.
- M1 disk: ring a waiting processor for owed steps instead of the boot
  processor (`OWED_DOORBELL`).
- Indirect branches `b9e131f` and system exits (`103cccd`, `69e09b7`) onto
  the merged branch, with full compiler and engine suites.

### Measured (cloud Xeon, same machine, before and after)

NVMe transfer bench (`nvme::owned::tests::transfer_cost`, release, best of
4 over 64 MiB, per MiB of 1 MiB guest transfers):

| Host command size | Commands | Write us | Read us |
|---|---|---|---|
| 128 KiB (base `0bc7857`) | 8 | 960 | 930 |
| 256 KiB | 4 | 1007 | 967 |
| 512 KiB | 2 | 1086 | 1001 |
| 1 MiB (`5329e69`) | 1 | 1161 | 1076 |

The model has no device latency and no step waits, so this shows only the
host copy and bookkeeping: 21 % slower at 1 MiB. `5329e69` is reverted
(`df0ea17`); the size returns as a boot option for an owner A/B.

Merge `a5d87ab` tests: engine 1975, soc 1629, storage 314, audio 405,
boot-core 1127 passed; compiler 1591 passed, 7 failed (the same FP16
seven as the base).

## 2026-11-21, session 1 continued: translation (M2)

Machine: the same cloud Xeon. Kernel numbers are compiled for the owner's
tier (AVX2, no AVX-512: `owner_target` in `kernel_costs`).

### Baseline (`kernel_costs`, ns per guest instruction, self-looping blocks)

global_counter 0.160, integer_alu 0.155 to 0.19 (noisy), global_read
0.185, bytes_ldrb_strb 0.214, list_chase 0.270, memcpy_unrolled 0.275,
memcpy_ldp_stp 0.355, stack_frame 0.392, neon_fmla 0.55 (5531 bytes).
A tight loop runs 2 to 6 G guest instructions/s per PE; the owner's boots
average about 0.3 G/s per PE, so the time is between blocks.

### Landed

| Commit | What | Before | After |
|---|---|---|---|
| `010b686` (from `b9e131f`) | predicted tail miss probes the jump cache | miss over 4 targets 12.54 ns, 64: 14.76, 1024: 18.17 | 8.36 (-33 %), 9.70 (-34 %), 11.47 (-37 %) |
| `fac17a2` (from `69e09b7`) | PAC key input lists are rings | open at a thread switch, 64 processes 2110 ns, 1024: 2075 | 235 (-89 %), 234 (-89 %) |

Tests: compiler 1591 passed (7 FP16 failures as before), engine 1975
passed. Clippy host and UEFI clean.

### Tried and dropped (measured)

- No NZCV store on a self-loop's back edge: -5 to -26 % per kernel as an
  upper bound, but not bit-exact: a fault in the next iteration before its
  flag setter, or the budget exit every 249 instructions, needs the word.
  Only loops whose head is pure arithmetic could take it.
- Reading RAX where the next instruction read the held register's frame
  slot: integer_alu median 0.177 -> 0.186 ns, others within 1 %. Memory
  renaming already forwards these; reverted.

## 2026-11-21, merge of the updated daily-driver (`27575a5`)

`performance/daily-driver` moved to `6fedcaf` (11 commits by the local
agents) while this branch worked. Merged into the upstream-merged branch:

- Disk: daily-driver's `7bc4fff` (owed steps to a waiting PE), `8cf55c7`
  (NVMe signal to the last PE's core; silent disk heard again) and
  `6fedcaf` (256 KiB commands, 4 MiB pool) win; this branch's `183fcc2`
  did the "heard again" part alone and gives way. The transfer bench
  here measured 256 KiB at +5 % host copy cost per MiB against 4x fewer
  commands (table above); 6fedcaf argues the in-flight bytes on the disk.
- Translations: `8c8041d`, `e692959`, `1748fca` (arena compaction) give
  way to upstream's on-demand pack, which has no arena.
- MFENCE removal and the jump probe were on both sides.

Tests on the merged tree: engine 1976, soc 1629, boot-core 1127,
driver-base 1120, audio 405, storage 316 passed; compiler 1593 passed,
7 FP16 failures (as before). fmt, host clippy and the UEFI clippy lanes
(engine, compiler, storage, boot-core, stela-boot) clean.

## 2026-11-21, goal: +40 % whole Mac (engine round trip)

Owner's boots average about 0.3 G guest instructions/s per PE while a
tight translated loop runs 2 to 6 G/s per PE: the time is between blocks.
Measured here: the engine's round trip for one native run
(`run_loop_costs`, the engine's fixed cost of an exit), callgrind for
deterministic host instruction counts (`RUN_LOOP_SERIAL=1`), perf
cpu-clock sampling for the parallel cases.

| Commit | Change | Serial host instructions (8 M runs) |
|---|---|---|
| base `a6e287c` | | 27.75 G |
| `bec611c` | variant id memo; recent contexts compare ids | 23.98 G |
| `a77fcc3` | ITLB salt shift by BSR, not software popcount | 23.41 G |
| `d647fe6` | missed-transfer classification sampled 1 in 8 | 20.43 G (-26 % total) |

ns per native run, cloud Xeon:

| Exit | PEs | Before | After |
|---|---|---|---|
| br | serial | 164 | 116 |
| br | 1 parallel | 189 | 133 to 142 |
| br | 2 | 213 | 157 |
| br | 4 | 220 | 158 |
| mrs+br | serial | 206 | 167 to 170 |
| mrs+br | 4 | 254 | 190 to 198 |

Also measured, not landed:
- Host CPU features (+popcnt,+bmi1,+bmi2,+lzcnt,+movbe) for the engine:
  br serial 136 -> 132 ns (-3 %): a build option candidate, as it would
  #UD on CPUs without them.
- Compile throughput of a 44 038-instruction arm64 corpus (mbedtls and
  Mesa C, clang -mcpu=apple-m1 -O2), owner tier: 4.2 us a block, 500 ns
  a guest instruction. An app's code compiles in well under a second, so
  app launch time is not compile bound.
- A self-loop back edge without its NZCV store: -5 to -26 % per kernel,
  not bit-exact (fault and budget exits need the word); dropped.

Tests: engine 1976 passed, compiler 1593 (7 FP16 as before); clippy host
and UEFI clean.

### Later the same day

| Commit | Change | Measured |
|---|---|---|
| `faa80d2` | `acknowledge` of a quiet control stores nothing (no locked exchange per exit) | br 2 PEs 157 -> 149 ns; race test 20/20 |
| `0853fc4` | `STELA_TARGET_FEATURES` build option (popcnt, bmi1/2, lzcnt, movbe) | br serial 136 -> 132 ns |
| `0b3a64b` | oracle manifest lists six opt-in benches it missed | xtask 210 passed |

Tests after `faa80d2`: compiler 1593 (7 FP16 as before), engine 1976;
clippy host and UEFI clean.

Looked at and left: PMU mirrors per entry (~67 host instructions a run),
the interrupt check (~100), the lower-half global page's second ITLB
probe (~70), `grace.pass`'s locked add (a required full barrier).

| `c3d325f` | a branch exit that missed its target fetches at once (skips schedule and device step) unless its budget ran out | br 4 PEs 151-167 -> 138-146 ns |
| `f870d59` | no write-back of an empty transition note per run | serial host instructions -1.3 % |

Engine round trip now, cloud Xeon: br serial 116-130 ns (164 at the
session's start), 4 PEs 138-146 ns (220); host instructions per serial
run about 2570 (3470).

## Owner checkpoints

The owner asks for the difference since their last question. Each line is
the state they were last told about.

- 2026-11-21 (asked 1): base `0bc7857`, now `5dc79ec`. Measured: NVMe 1 MiB
  commands 21 % slower per MiB on the host (reverted). Guest instructions
  per second: unchanged by this session (no translator or engine change
  beyond upstream's merge); the owner baseline stays session 3, 4.98 G/s
  peak, 2.42 G/s median.
- 2026-11-21 (asked 2): now `a410791`. Since checkpoint 1: indirect
  branch misses 33 to 37 % faster, PAC key switches 89 % faster at 64
  processes and more. The owner's last measured rate is session 3's
  (2.42 G/s median, 4.98 G/s peak); the next boot's `GUEST PERFORMANCE`
  line is the new figure.
- 2026-11-21 (asked 3, goal set): base `a6e287c`. Engine round trip of a
  native run 164 -> 116 ns serial (-29 %), 220 -> 158 ns on 4 PEs (-28 %).
