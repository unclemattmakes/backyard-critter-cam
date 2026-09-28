# The host hangs, and the rig is the best trigger it has ever had

**Status:** open. **The PSU was replaced on 2026-09-13 and the box hung again 11 minutes after
the rig came back up.** Power delivery *at the PSU* is effectively excluded; see *Results*.
**Machine:** OLVR (ASUS PRIME X299-DELUXE, i7-7820X, RTX 3070, 64 GB, BIOS 1401 / 2018-05-09).
**Last updated:** 2026-09-23 — see *2026-09-23: interval result*. Earlier: 2026-09-13 23:30 — supersedes both the 2026-08-26 reading and the earlier
2026-09-13 one. See *The recorder's blind spot*, which is now the most important section here.

This is not a bug in the rig. It is written down here because the rig is what surfaced it, the
rig is what keeps dying of it, and the next person to lose a night of raccoons to a black screen
should not have to re-derive any of it.

## What happens

The machine hard-hangs under load. On 2026-08-24 it was found frozen on the Windows lock screen
with the picture intact and the keyboard and mouse dead — the display controller was still
scanning out a frame nothing behind it was alive to update.

Losses since the rig moved onto this box:

| | rig started | box died | rig had been up |
|---|---|---|---|
| 2026-08-23 | 20:52 | 23:02:10 | 2h 10m |
| 2026-08-24 | 13:04 | 13:10:31 | 6m |
| 2026-08-28 | 18:44:44 | 18:46:28 | **104 s** |
| 2026-09-06 | 16:49:15 | 16:49:30 | **15 s** |
| 2026-09-07 | 17:22:09 | 22:36:51 | 5h 15m |
| 2026-09-13 | 22:38:35 | 22:49:28 | **11m** — *new PSU, 140 W cap, mid-visit* |
| 2026-09-14 | 23:28:36 (prev day) | 07:27:38 | **8h** — *new PSU, mid-visit* |
| 2026-09-15 | 21:16:16 (prev day) | ~12:04 | ~15h — *3.0 s detector interval* |
| 2026-09-15 | 18:09:29 | ~18:11 | **~90 s** — *naming never reached ready; frozen 6 days* |
| 2026-09-21 | 22:21:37 | ~22:23:37 | **~2 min** — *mid-visit, naming never reached ready* |
| 2026-09-23 | 13:33:41 | 13:35–13:38 | **~2–4 min** — *naming still loading* |

The 2026-09-14 event is the same shape as 09-13 at a different scale, and it corrects any
impression that the new PSU made things dramatically worse: eight hours, not eleven minutes. It
died mid-visit again — two clips encoding (380 and 407 frames), and **24 detector wakes in the
25 seconds before the last sample**, which is `detector_min_interval_s` saturating. The box then
sat frozen for **12.5 hours**, recovered by power cycle at 19:55, and came back to the lock
screen, so the logon-only tasks stayed down for another hour until someone logged in at 20:56.

The 2026-09-13 event is the best-documented one and the most important. It happened on a
brand-new PSU with the GPU capped at 140 W, eleven minutes after the rig restarted, with an
animal in frame and detections firing once or twice a second. It was found frozen on the Windows
login screen — the session had been locked by hand beforehand, which is incidental: no
screensaver timeout and no `InactivityTimeoutSecs` are set, and the display timeout is 60
minutes, so nothing auto-locked or powered anything down. Recovered by power cycle at 23:25.

Two earlier events are ambiguous and are not counted as hangs: 2026-09-08 17:15 (the rig had been
stopped by hand) and 2026-09-11 11:38 (rig not running, no telemetry). The three Kernel-Power 41
records on 2026-09-13 between 21:46 and 22:04 were the PSU replacement, not failures.

## It is the box, not the rig, and not new

OLVR has ended this way **16 times since 2025-10**, and **13 of those predate the rig ever
running on it**. The rig is a trigger, not a cause. The same code ran flawlessly across fifteen
sessions on the old laptop (RTX 5050) on 2026-08-20/21/22.

**What the 13 earlier events probably have in common with the rig is the GPU, not the software.**
Before the rig, the heavy workload on this box was VR — the Oculus runtime, SteamVR and Virtual
Desktop are all installed, and the machine is named for it. VR pins the GPU exactly as the rig's
detector bursts do. Read that way, the whole 16-event history unifies under *sustained or bursty
GPU load* rather than under anything specific to this project, which is also the reading that
best explains why a GPU power cap bought two orders of magnitude.

The application has never been implicated and still isn't: **zero tracebacks** across the current
and all five rotated `backyard_cam.log` files, and only **four** rigwatch restarts in three weeks
(2026-08-21, 08-24, 09-08, 09-13), two of which merely follow a host restart. When the rig
"crashes," the rig is not what crashed.

## What the absence of evidence tells us

Across all events there is **no bugcheck, no crash dump, no WHEA record and no TDR** — and dumps
are correctly configured (`CrashDumpEnabled=7`, dump path set, `AutoReboot=1`). That combination
is the finding, not a dead end:

- A bluescreen writes a dump. A display-driver fault writes a TDR event. A correctable hardware
  error writes a WHEA record. Getting **none** of them points at a CPU that stopped retiring
  instructions rather than any software fault.
- Ruled out along the way: **storage** (C: is a healthy Samsung 850 EVO with no controller resets;
  the 7 WHEA errors on file decode to STORPORT faults on the secondary WD10EZEX drive and stopped
  in 2025-05), **GPU driver** (zero TDRs ever, and the hangs predate the current driver by a
  year), and **ASUS AI Suite's kernel drivers** (the usual X299 suspect — not installed here).
- The 10,431 `Kernel-WHEA/Operational` entries are boot-time initialisation noise, not errors.

## The recorder's blind spot

**`hosthealth.ps1` samples every 5 s. The workload that kills this box is millisecond-scale.**

Read naively, every freeze looks like it happened at idle — GPU at 16.9–17.0 W, P8, 210 MHz, 0%
utilisation, PCIe gen 1, CPU under 25%. That reading is an artifact and it is wrong. MegaDetector
on a 3070 is roughly 30 ms per frame: the GPU wakes P8 to P2, spikes to 1815 MHz and gen 3, and is
back asleep long before the next poll. On 2026-09-07 the app log records detections firing about
once a second while the recorder, sampling between them, wrote `17.00 W / P8 / 210 MHz` twice in
a row and then stopped forever.

The single exception is a sample that happened to land inside a wake — 2026-09-06 16:49:15,
`51.33 W / 1815 MHz / P2 / gen 3` — fifteen seconds before the machine died.

### Sampling at 1 Hz does not fix it — proven 2026-09-13

A second recorder was run at `-Interval 1` during the 2026-09-13 freeze. Put its output beside
the application log for the same seconds:

| clock | `backyard_cam.log` | `hosthealth-1s` |
|---|---|---|
| 22:49:12 | animal 0.69, animal 0.41 | 21.29 W, P8, 210 MHz, gen 1 |
| 22:49:13 | animal 0.54, animal 0.33 | 21.24 W, P8, 210 MHz, gen 1 |
| 22:49:14 | animal 0.59, animal 0.30 | 21.40 W, P8, 210 MHz, gen 1 |
| 22:49:15 | animal 0.60 | 21.20 W, P8, 210 MHz, gen 1 |
| 22:49:17 | animal 0.48 | 21.21 W, P8, 210 MHz, gen 1 |
| 22:49:19 | animal 0.43 | 21.29 W, P8, 210 MHz, gen 1 |
| 22:49:28 | — | 20.46 W, P8, 210 MHz, gen 1 — **last sample** |

The rig was detecting one to two animals per second, writing crops and encoding a clip, and the
recorder reported a GPU sitting at minimum clock and minimum power for forty consecutive seconds.
Both are accurate. `nvidia-smi` returns the *instantaneous* state at poll time, and between two
30 ms inferences that state is always idle.

**Conclusion: polling `nvidia-smi` cannot observe this workload at any practical rate, and should
not be used to reason about what the hardware was doing.** The recorder's remaining value is
establishing *that* the machine stopped, *when*, and what memory, disk queue and process
footprint looked like. For power and clock behaviour it is worse than useless, because it returns
a confident number that is systematically wrong.

Every previous "it died at idle" reading in this investigation came from this artifact. The box
has not been observed to die at idle even once; on 2026-09-13 it died with an animal in frame.

## What the failures actually look like

Two shapes, one underlying moment.

**Death during startup** (2026-08-28, 2026-09-06). The rig's first two minutes are the heaviest
concurrent thing it ever does: MegaDetector loading onto the GPU while the species model warms on
the CPU — "~1-2 min" per the startup banner. On 08-28 the CPU sat at 51–54% from 18:45:52 to
18:46:18, squarely inside that warm-up, and the box was gone at 18:46:28, 104 seconds after
`Watching for critters`. On 09-06 it lasted fifteen seconds.

**Death mid-burst** (2026-09-07). Fourteen detections between 22:36:23 and 22:36:43, two clips
encoding, crops landing about once a second. Last crop written 22:36:51. Gone.

The common factor is **GPU inference and CPU AVX-512 classification firing at the same time**,
not sustained load. Sustained load is survivable: the box ran 8.8 days unbroken with the rig up
(see *Results*), and `tools/memcheck.py` saturated memory bandwidth for 17.5 minutes on top of
the live rig without a flinch.

Three details corroborate the burst reading:

- **The crops stop at `animal`.** All 303 crops from 2026-09-07 carry the detector's generic
  `animal` label; not one reached a species. The crop is written by the GPU stage, the species
  comes from the CPU stage, and the machine dies in the handoff.
- **A clip truncated on an allocation boundary.** `clips/glass_door_cam/2026-09-07/2026-09-07T22-36-47-966.mp4`
  is 1,048,624 bytes — 48 bytes past exactly 1 MiB. That is a hard power loss mid-write, not a
  clean stop.
- **`logs/naming.log` begins with 371 NUL bytes.** Cached writes the freeze discarded.

## Ranked hypotheses

**The single most informative constraint is an asymmetry:** capping the GPU to 140 W changed the
failure rate by roughly two orders of magnitude (5.5 hours uncapped, 8.8 days capped), while
replacing the PSU changed nothing at all (11 minutes). Whatever is marginal is therefore
**sensitive to the GPU's power behaviour but not to the PSU that feeds it** — which points
downstream of the PSU, or at transition *magnitude* rather than absolute draw.

1. **Transient power delivery downstream of the PSU.** The GPU's own power path — PCIe power
   cables and connectors, the card's VRM, slot delivery — or the board's 12 V distribution and
   CPU VRM, where a GPU wake (P8 to P2, ~20 W to 52 W+ in well under a frame time) lands on top
   of the CPU entering AVX-512. Fits every observation: hard hang, no error record, correlated
   with detection bursts, responsive to the GPU cap, unresponsive to a new PSU.
   **The new PSU shipped with its own PCIe cables and those were the ones fitted** (confirmed
   2026-09-13), so supply *and* cabling are both new hardware. Everything from the wall to the
   card's power connectors has now been replaced, and the fault survived it. What remains on this
   branch is the **card's own VRM, the PCIe slot and the board's power delivery** — nothing
   upstream of them.
   A second reading of the cap's effect is equally live: capping also shrinks every clock and
   voltage excursion the card makes, so the benefit may be about transition magnitude rather
   than power, which would implicate the board and CPU VRMs by a different route.
2. **Stale firmware / microcode.** BIOS 1401 ships microcode `0x2000043`; Windows patches it only
   to `0x200005E`. Both are 2018-vintage on a part that received years of revisions afterwards.
   AVX-512 license-level transitions are the sharpest VRM event this CPU can generate, and since
   the minimum processor state came off 100% the CPU now makes them from a much lower clock — a
   wider excursion than it ever had to make before. VBS and the Hyper-V hypervisor are also
   active, so everything runs a layer down on that microcode.
3. **Thermal — CPU or VRM.** The box was dormant 2026-05-26 to 08-20 and came back in August heat.
   Skylake-X under AVX-512 is about as hot as Intel ever shipped. Weakened but not retired: GPU
   temperature never exceeded 60 °C across three weeks of sampling, and the CPU and VRM sensors
   that would actually settle this are not exposed. See the blind spot below.
4. **Memory instability.** Lowered by `tools/memcheck.py` on 2026-08-26 — 37.4 GiB swept for 17.5
   minutes, two passes of seven patterns including address-in-address and a 30 s retention soak,
   zero bad words, write bandwidth holding ~20 GiB/s throughout. Not retired: a userspace test
   cannot see kernel- or firmware-held memory and cannot defeat the cache. MemTest86 is still
   what clears the RAM.

## What is instrumented now

Because a hang leaves nothing behind, the evidence has to be gathered *before* it.

- **GPU board power capped to 140 W** (from 270 W). `nvidia-smi -pl` does not persist across a
  reboot on Windows, so run `tools/apply_gpu_cap.ps1` **once from an elevated PowerShell** — it
  applies the cap and registers a SYSTEM boot task that reapplies it, with retries, because at
  startup the task will otherwise beat the display driver. `-Remove` undoes both. Verified
  working: the cap was back at 140 W on its own after the 2026-09-13 PSU swap.
- **CPU minimum processor state dropped 100% to 5%.** Note this cuts both ways — it lets the CPU
  idle down, and it widens every frequency excursion it subsequently has to make. See hypothesis 2.
- **`tools/hosthealth.ps1`** samples every 5 s into `logs/hosthealth/hosthealth-<date>.csv`,
  opened `WriteThrough` and flushed per sample so the last rows are on the device rather than in a
  cache a freeze would discard. Scheduled as **"Backyard critter-cam host telemetry"**, started at
  logon. **Run it with `-Interval 1` during a deliberate test** — 5 s is too coarse to resolve the
  edge.
- **`rigwatch.py`** — **"Backyard critter-cam rigwatch"**, every 5 minutes. Covers the app dying,
  not the host hanging; a frozen machine still needs a physical power cycle.

**A known blind spot:** CPU package and VRM temperatures are not exposed to userspace on this
board, and those are exactly what hypotheses 2 and 3 turn on. Re-confirmed 2026-09-13:
`MSAcpi_ThermalZoneTemperature` returns nothing at all and `Win32_TemperatureProbe` returns empty
readings, so no ordinary program can see them — reading them needs a tool that ships its own
kernel driver.

**HWiNFO64 8.52 portable is installed at `C:\Users\MatthewS\HWiNFO64`** (downloaded 2026-09-13,
Authenticode verified: EV certificate issued to REALiX, s.r.o., DigiCert-timestamped). Portable
rather than installed, so removing it is deleting the folder. It must be launched **as
Administrator** or it cannot load its driver and the sensors that matter stay invisible. Run it
sensors-only, set the polling period to 200–500 ms, and log to CSV.

Unverified and worth checking before trusting it: whether HWiNFO flushes its CSV per row. If it
buffers, a hard hang discards exactly the rows this whole exercise exists to capture — the trap
`hosthealth.ps1` uses `WriteThrough` to avoid. Its shared-memory interface may also be Pro-only.

**Every critter-cam scheduled task is "run only when logged on."** A reboot that stops at the lock
screen leaves the yard dark and the recorder silent until someone logs in. This has already cost a
24.5-hour outage once and explains several gaps in the CSV series.

## Results so far

**The 140 W cap helped enormously and did not fix it.** Uncapped, the rig managed 5.5 hours and
then 6 minutes. Capped, the box ran **8.8 days unbroken** — 2026-08-28 21:52 to 2026-09-06 16:49,
roughly 17,000 samples a day with no gap, no reboot and no Kernel-Power 41, rig up throughout.
Board power topped out at 139.48 W against the cap, so the limit was genuinely binding.

Then it died again on 09-06 and 09-07. Capping reduced the rate by something like two orders of
magnitude; it did not remove the failure. Read honestly, **three variables have now moved** — the
GPU cap, the CPU minimum state, and the PSU — and none of them has been isolated.

**The 2026-08-26 entry claimed the cap was holding at 49.7 hours and reasoned from the CSV that
the freezes happened at idle. Both readings are superseded** — the first by the 09-06 and 09-07
events, the second by the sampling artifact described above.

**2026-09-13 — the PSU did not fix it.** New supply in at ~22:04. Rig restarted 22:38:35, ran
normally through a real visit, and the box hard-hung at 22:49:28 — eleven minutes — with the GPU
still capped at 140 W. Recovered by power cycle at 23:25. This is as close to excluding the PSU
as a single event gets: a new supply, an 850 W unit that was never near its capacity, a GPU
limited to roughly half its nameplate, and a modest detection workload.

**2026-09-13 — `clipmotion --redo` is not a faithful reproducer.** The idea was to replay 4,028
saved clips through MegaDetector to trigger the fault on demand. Measured at 1 Hz, the replay
runs as a *steady* load — P5, ~31 W, 40% utilisation, CPU ~11% — because it is decode-bound and
the GPU never returns to idle between inferences. It exercises sustained load, which this box
demonstrably survives, and not the P8-to-P2 burst pattern that accompanies every real failure.
Worth keeping as a soak test; do not mistake it for a reproducer. The live rig remains the only
known way to provoke the fault, and on 2026-09-13 it took eleven minutes.

## The collateral damage: an 8.6-day naming outage

The 2026-08-28 freeze did not only cost a night. The rig came back at 22:02, the naming helper
started, wrote `ready` to `.naming_status.json` at about 22:07 — and then died and never wrote
again. It stayed dead for **8.6 days**, until 2026-09-06, with the status file still reading
`ready` and the dashboard pill still green. No crops named for over a week, which also means no
visits, no re-ID templates and no nightly eval.

The mechanism is the one the NULs point at: a hard hang leaves NUL-filled files behind, and
NUL-filled crops kill the helper on every start. **A freeze is not a one-night outage; it can
silently void the corpus until someone notices.**

`rigwatch.py` now carries a naming-health check (`NAMING_STALE_S = 900`) that logs `NAMING IS
STALE` rather than restarting anything — the rig supervises its own naming child. It is what
finally surfaced this on 09-06. **That work is uncommitted.** Commit it.

## 2026-09-23: interval result, and what changed

**The detector-interval experiment is answered: no.** At 3.0 s the box died after ~15 h, then
~90 s, ~2 min and ~2–4 min. That is not triple the 1.0 s baselines, so *detector wake rate* is not
the trigger. What the last three share is **timing**: all died inside the first four minutes,
before the naming helper ever wrote `ready`, meaning BioCLIP was loading and warming up on the
CPU while the detector was already live on the GPU.

The CPU half turns out to be heavier than the doc assumed. The venv's torch
(`2.12.0+cu130`) reports `get_cpu_capability() == "AVX512"` with 8 threads, so the naming
helper drives **all eight cores into AVX-512** during warm-up. That is the sharpest current step
this CPU can take, and it lines up with hypotheses 1 and 2.

Two changes, both 2026-09-23:

- **The naming child is capped to AVX2 on 4 threads** (`Config.classify_cpu_isa`,
  `classify_cpu_threads`; applied through the env in `backyard_cam._naming_env`). This is a
  mitigation and also the next experiment. If the startup deaths stop, the CPU-side AVX-512 burst
  is the trigger, and the BIOS update (microcode, and checking MultiCore Enhancement / AVX offset
  on this 2018 ASUS firmware) becomes the real fix. If they don't stop, look at the GPU card or slot.
- **`python rigwatch.py --hold`** keeps the watchdog from restarting the rig, across reboots,
  until `--release`. Before this, `.rig_pause` aged out at every crash-reboot, and rigwatch started
  the rig straight back into the hang (09-15 18:03, 09-23 13:33), so the machine could not be
  kept quiet for debugging. **A hold is set as of 2026-09-23 13:42.**

## The detector-interval experiment — ran 2026-09-14 21:16 to 2026-09-23 (see above)

**Someone has to come back and read this.** On 2026-09-14 `config.py`'s
`detector_min_interval_s` was raised **1.0 → 3.0** (and `box_display_ttl_s` with it, so the live
preview does not sit boxless for two seconds in three). The rig was restarted at 21:16:16 to pick
it up.

It is a mitigation and a dose-response test at the same time. Every detector run drags the GPU
P8 → P2 for roughly 30 ms, and those transitions are the best correlate of the fault anyone has
found. Cutting the wake rate to a third should, if the transitions are really the trigger, buy
roughly triple the time-to-failure.

| outcome | reading |
|---|---|
| time-to-failure roughly triples | strongest evidence yet that **transitions** are the trigger, not sustained load or temperature |
| time-to-failure unchanged | the transition hypothesis is **wrong**; stop chasing it and look at firmware, VRM or the card |
| no more hangs at all | suggestive but not conclusive — the box once went 8.8 days on its own |

Baselines to measure against, all on the new PSU: **11 minutes** (09-13) and **8 hours** (09-14),
against a pre-cap record of 5.5 hours then 6 minutes, and a best-ever capped run of 8.8 days.
Note the variance is enormous, so a single long run proves very little — this needs several
failures before the comparison means anything.

The cost is about a third as many crops per visit, which thins re-ID still density. Clips are
untouched (they record at full camera rate), so `clipmotion`'s gait and behaviour work is
unaffected. **Revert to 1.0 once the host is trustworthy.**

## Instrumentation status: still blind

As of 2026-09-14, **three consecutive freezes have gone uninstrumented.** HWiNFO64 was installed
and launched on 09-14, but logging was never started and the process did not survive the reboot,
so no CSV exists. The CPU package power and VRM temperatures that hypotheses 2 and 3 both turn on
have still never been observed on this machine, not once.

Whatever else happens, **verify logging is actually writing rows before trusting that it is** —
the pattern here is a recorder that everyone believes is running and is not, which is the same
failure that let the naming helper sit dead for 8.6 days behind a status file reading `ready`.

## Next actions, by information gained per unit of effort

1. **HWiNFO64, sensors-only, CSV logging, poll interval 200 ms or faster.** This is no longer a
   nice-to-have. `nvidia-smi` polling has been shown incapable of seeing this workload at all, so
   right now the investigation has *no* instrument that can observe the last second before a
   hang. HWiNFO64 reads CPU package power, per-phase VRM temperatures and GPU power through the
   driver rather than by spawning a process, which is both faster and the one blind spot that
   hypotheses 1 through 3 all turn on. Everything below is guesswork until this exists.
2. **Reseat the card, and move it to a different PCIe x16 slot.** Free, and it is now the cheapest
   live test on the leading branch: the supply and cables are new, so if slot-side power delivery
   or the slot contacts are marginal, a different slot either fixes it or rules the slot out.
   Reseat the 8-pin connectors at both ends while the case is open.
   **There is somewhere to move it.** SMBIOS reports six slots, and only one is occupied:

   | slot | state |
   |---|---|
   | PCIEX16_1 | **in use** — the RTX 3070, at PCI bus `65:00.0` |
   | PCIEX16_2 | free |
   | PCIEX16_3 | free |
   | PCIEX16_4 | free |
   | PCIEX1_1, PCIEX1_2 | free (x1, not usable for the card) |

   The 7820X has 28 PCIe lanes, so the x16 slots do not all get full width — check the manual's
   slot table for which pair the CPU feeds at x16, and confirm afterwards with
   `nvidia-smi --query-gpu=pcie.link.width.max --format=csv`, which currently reports 16.
3. **Update the BIOS from 1401.** Eight years of microcode in one step, aimed squarely at
   hypothesis 2, and cheap. Ordering trap: it requires a reboot, so `tools/apply_gpu_cap.ps1`
   must have registered its boot task first or the machine comes back at 270 W and the next
   result is worthless.
4. **Borrow or fit a different GPU for a few days.** If the box stops hanging with another card
   in the slot, the fault is the 3070 or its power path and the search is over. Note the i7-7820X
   has no integrated graphics, so this needs an actual spare card, and it is the single most
   decisive test available. Remove the power cap for the duration or the comparison is not clean.
5. **MemTest86, overnight**, if you want the RAM genuinely cleared rather than merely doubted.
6. **Dust it out; inspect the CPU cooler.** If the AIO dates to the build, its pump is at end of life.

**Not worth repeating:** raising the GPU cap to isolate transients from the CPU-idle change. That
test was designed when the PSU was the leading theory; with the PSU excluded it now only risks
returning the box to its 5.5-hour failure rate in exchange for very little.
