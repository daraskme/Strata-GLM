"""Single-GPU GLM server with memory guards and optional Mang-AI GPU queue.

Never stops other applications. The owned process group is released on exit.
Default is a read-only plan; --run explicitly starts the large model.
"""
import argparse
import ctypes
from contextlib import nullcontext, contextmanager, ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30


def resources():
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total,pcie.link.gen.current,pcie.link.width.current,utilization.gpu", "--format=csv,noheader,nounits"],
                         text=True, capture_output=True, check=True, timeout=10)
    free, total, gen, width, util = map(float, gpu.stdout.splitlines()[0].split(","))
    return {"ram_available_gib": int(mem["MemAvailable"].split()[0])/1024**2,
            "vram_free_gib": free/1024, "vram_total_gib": total/1024,
            "pcie_gen": int(gen), "pcie_width": int(width), "gpu_util_percent": util}


@contextmanager
def max_performance(enabled):
    """Temporary NVML PowerMizer hint, verified through the same live client.

    NVIDIA go-nvml nvml.h: nvmlDevicePowerMizerModes_v1_t (three uints,
    no version field). No clock locks, driver reloads or PCIe register writes.
    """
    if not enabled:
        yield
        return
    class Modes(ctypes.Structure):
        _fields_ = [("currentMode", ctypes.c_uint), ("mode", ctypes.c_uint),
                    ("supportedPowerMizerModes", ctypes.c_uint)]
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    handle = ctypes.c_void_p()
    signatures = {"nvmlInit_v2": [], "nvmlShutdown": [],
                  "nvmlDeviceGetHandleByIndex_v2": [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
                  "nvmlDeviceGetPowerMizerMode_v1": [ctypes.c_void_p, ctypes.POINTER(Modes)],
                  "nvmlDeviceSetPowerMizerMode_v1": [ctypes.c_void_p, ctypes.POINTER(Modes)]}
    for name, args in signatures.items():
        fn = getattr(nvml, name)
        fn.argtypes, fn.restype = args, ctypes.c_int
    nvml.nvmlErrorString.argtypes = [ctypes.c_int]
    nvml.nvmlErrorString.restype = ctypes.c_char_p

    def checked(rc):
        if rc:
            raise RuntimeError("NVML: " + nvml.nvmlErrorString(rc).decode())

    def get():
        data = Modes()
        checked(nvml.nvmlDeviceGetPowerMizerMode_v1(handle, ctypes.byref(data)))
        return data

    def set_mode(mode):
        data = Modes(mode=mode)
        checked(nvml.nvmlDeviceSetPowerMizerMode_v1(handle, ctypes.byref(data)))

    checked(nvml.nvmlInit_v2())
    old = None
    try:
        checked(nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)))
        data = get()
        if not data.supportedPowerMizerModes & (1 << 1):
            raise RuntimeError("maximum-performance mode is not supported")
        old = data.currentMode
        set_mode(1)
        if get().currentMode != 1:
            raise RuntimeError("driver did not retain maximum-performance mode; retry without --prefer-max-performance")
        print(f"PowerMizer {old} -> 1 (終了時に復元)", flush=True)
        yield
    finally:
        try:
            if old is not None:
                set_mode(old)
                if get().currentMode != old:
                    raise RuntimeError(f"PowerMizer restoration failed; expected {old}")
                print(f"PowerMizerを{old}へ復元しました", flush=True)
        finally:
            checked(nvml.nvmlShutdown())


def resident_ready(url):
    if not url:
        return True
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except OSError:
        return False


def validate_external_lease(root):
    """Validate the inherited descriptor without acquiring a second GPU lease."""
    try:
        fd = int(os.environ["MANGAI_GPU_LEASE_FD"])
        if fd < 3:
            raise ValueError("invalid lease descriptor")
        held = os.fstat(fd)
        lock_path = root.resolve() / "work/gpu/heavy.lock"
        expected = lock_path.stat()
        if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("lease descriptor belongs to a different queue")
        with lock_path.open("rb") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return fd
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise ValueError("inherited GPU lease is not locked")
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(f"external GPU lease requires the broker's locked descriptor: {error}") from error


def parse_cpu_plan(value):
    """CPU expert count for each possible RAM-tier miss count, zero through eight."""
    if len(value) != 9 or any(char not in "012345678" or int(char) > count
                              for count, char in enumerate(value)):
        raise argparse.ArgumentTypeError("CPU plan must have 9 ASCII digits; digit f must be between 0 and f")
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", required=True, type=Path)
    p.add_argument("--context", type=int, choices=[32768, 65536, 131072, 262144], default=131072)
    p.add_argument("--port", type=int, default=1243)
    p.add_argument("--model-name", default="glm-5.3-flash-orcarouter-q4-mangai", help="API model identity; use a distinct ID for each checkpoint")
    p.add_argument("--vram-cap-gib", type=float, default=90)
    p.add_argument("--ram-headroom-gib", type=float, default=12)
    p.add_argument("--expert-pool-gib", type=float, help="upper bound for the GPU expert pool in controlled comparisons")
    p.add_argument("--ram-tier-gib", type=float, help="fixed pinned expert RAM budget; retains headroom plus 4 GiB setup allowance")
    p.add_argument("--min-ram-gib", type=float, default=94)
    p.add_argument("--min-vram-gib", type=float, default=80)
    p.add_argument("--cpu-lane", choices=["auto", "off"], default="auto")
    p.add_argument("--cpu-plan", type=parse_cpu_plan, help="explicit CPU counts for 0..8 RAM experts, e.g. 001122334; omit for measured auto split")
    p.add_argument("--cpu-threads", type=int, help="explicit CPU lane threads for controlled comparisons")
    p.add_argument("--cpu-spin-us", type=int, default=20000, help="CPU lane idle spin before sleeping (0..20000 microseconds)")
    p.add_argument("--cpu-affinity", help="Linux CPU list for this GLM process only, e.g. 0-15 on this i9-12900KS")
    p.add_argument("--uniform-expert-slots", action="store_true", help="disable profile-based layer slot allocation")
    p.add_argument("--usage-expert-slots", action="store_true", help="allocate layer slots by saved usage per byte (experimental)")
    p.add_argument("--usage-profile", type=Path, help="expert usage seed; benchmark runs copy it instead of modifying it")
    p.add_argument("--prefetch-experts", type=int, choices=range(3), default=0, help="predictively copy 0..2 next-layer RAM experts to VRAM; experimental")
    p.add_argument("--lookahead-layers", type=int, choices=range(5), default=0, help="predict SSD expert reads for 0..4 following layers; experimental")
    p.add_argument("--prefill-mb", type=int, default=8192)
    p.add_argument("--prefill-chunk", type=int, default=16384)
    p.add_argument("--prefill-min", type=int, default=256, help="use fast token path for shorter inputs on this CPU/PCIe profile")
    p.add_argument("--mangai-root", type=Path)
    p.add_argument("--external-gpu-lease", action="store_true", help="broker owns the inherited GPU lease; requires --mangai-root")
    p.add_argument("--thinking-budget", type=int, default=0, help="artificial reasoning token cap; 0 keeps native reasoning within the request output limit")
    p.add_argument("--trace-experts", action="store_true", help="record expert routes for tier analysis; may produce large logs")
    p.add_argument("--profile-gpu-after", type=int, help="diagnostic CUDA interval CSV after skipping N fast passes; adds overhead, disabled by default")
    p.add_argument("--ram-policy", choices=["protected-lfu", "lfu", "lru"], default="protected-lfu")
    p.add_argument("--ram-protect-tokens", type=int, default=64)
    p.add_argument("--resident-health", default="http://127.0.0.1:1240/health")
    p.add_argument("--benchmark", action="store_true", help="disable conversation prefix reuse")
    p.add_argument("--prefer-max-performance", action="store_true", help="temporarily request NVIDIA maximum performance; restore on exit")
    p.add_argument("--run", action="store_true")
    a = p.parse_args(argv)
    if not 0 <= a.thinking_budget < a.context:
        p.error("thinking budget must be 0 (uncapped) or less than context")
    if not 0 <= a.ram_protect_tokens <= 4096:
        p.error("RAM protection must be between 0 and 4096 tokens")
    if a.profile_gpu_after is not None and a.profile_gpu_after < 0:
        p.error("GPU profile skip must be nonnegative")
    if a.external_gpu_lease and not a.mangai_root:
        p.error("external GPU lease requires --mangai-root")
    if a.uniform_expert_slots and a.usage_expert_slots:
        p.error("choose uniform or usage-based expert slots, not both")
    if a.cpu_plan is not None and a.cpu_lane == "off":
        p.error("CPU plan requires the CPU lane; use --cpu-lane off alone")
    if a.cpu_threads is not None and (not 1 <= a.cpu_threads <= 64 or a.cpu_lane == "off"):
        p.error("CPU threads must be 1..64 and require the CPU lane")
    if not 0 <= a.cpu_spin_us <= 20000:
        p.error("CPU spin must be 0..20000 microseconds")
    affinity = None
    if a.cpu_affinity:
        try:
            affinity = set()
            allowed_cpus = os.sched_getaffinity(0)
            for item in a.cpu_affinity.split(","):
                ends = list(map(int, item.split("-")))
                if any(cpu < 0 or cpu > max(allowed_cpus) for cpu in ends):
                    raise ValueError()
                if len(ends) == 1:
                    affinity.add(ends[0])
                elif len(ends) == 2 and ends[0] <= ends[1]:
                    affinity.update(range(ends[0], ends[1]+1))
                else:
                    raise ValueError()
            if not affinity or not affinity <= allowed_cpus:
                raise ValueError()
        except (ValueError, AttributeError):
            p.error("CPU affinity must be a nonempty subset of this process's allowed Linux CPUs")
    if a.ram_headroom_gib < 12 or a.vram_cap_gib <= 0:
        p.error("RAM headroom must be >=12 GiB and VRAM cap must be positive")
    for name, value in (("expert pool", a.expert_pool_gib), ("RAM tier", a.ram_tier_gib)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            p.error(f"{name} budget must be finite and positive")
    if a.expert_pool_gib is not None and a.expert_pool_gib > a.vram_cap_gib:
        p.error("expert pool budget must not exceed the total VRAM cap")
    if not 1024 <= a.prefill_mb <= 8192 or not 256 <= a.prefill_chunk <= 16384 or not 1 <= a.prefill_min <= a.prefill_chunk:
        p.error("prefill budget must be 1024..8192 MiB, chunk 256..16384 tokens")
    pack = a.pack.resolve()
    usage_seed = (a.usage_profile or pack / "expert_usage.txt").resolve()
    if a.usage_profile and not usage_seed.is_file():
        p.error(f"missing usage seed {usage_seed}")
    exe = ROOT / "build/strata"
    for path in [exe, pack / "tokenizer/tokenizer.json", pack / "tokenizer/chat_template.jinja"]:
        if not path.is_file():
            p.error(f"missing {path}")
    info = resources()
    required_ram = max(a.min_ram_gib, (a.ram_tier_gib + a.ram_headroom_gib + 4) if a.ram_tier_gib is not None else 0)
    if a.ram_tier_gib is not None and info["ram_available_gib"] < required_ram:
        p.error("fixed RAM tier would consume the configured headroom and setup allowance")
    if a.vram_cap_gib > info["vram_total_gib"] - 4:
        p.error("VRAM cap must leave at least 4 GiB outside the cap")
    plan = {"pack": str(pack), "model_name": a.model_name, "context": a.context, "port": a.port, "resources": info,
            "resident_ready": resident_ready(a.resident_health), "cpu_lane": a.cpu_lane,
            "cpu_threads": a.cpu_threads, "cpu_spin_us": a.cpu_spin_us, "cpu_plan": a.cpu_plan,
            "cpu_affinity": sorted(affinity) if affinity else None,
            "uniform_expert_slots": a.uniform_expert_slots,
            "usage_expert_slots": a.usage_expert_slots,
            "usage_seed": str(usage_seed),
            "vram_cap_gib": a.vram_cap_gib, "ram_headroom_gib": a.ram_headroom_gib,
            "expert_pool_gib": a.expert_pool_gib, "ram_tier_gib": a.ram_tier_gib,
            "prefill_mb": a.prefill_mb, "prefill_chunk": a.prefill_chunk, "prefill_min": a.prefill_min,
            "prefetch_experts": a.prefetch_experts, "lookahead_layers": a.lookahead_layers,
            "prefix_reuse": not a.benchmark, "prefer_max_performance": a.prefer_max_performance,
            "thinking_budget": a.thinking_budget, "external_gpu_lease": a.external_gpu_lease,
            "trace_experts": a.trace_experts, "profile_gpu_after": a.profile_gpu_after, "ram_policy": a.ram_policy,
            "ram_protect_tokens": a.ram_protect_tokens}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not a.run:
        return
    if a.external_gpu_lease:
        try:
            validate_external_lease(a.mangai_root)
        except ValueError as error:
            p.error(str(error))
    if not plan["resident_ready"]:
        p.error("resident A1 is not ready; start it first")
    if info["ram_available_gib"] < required_ram or info["vram_free_gib"] < a.min_vram_gib:
        p.error("insufficient free memory; close VMs/other large models yourself and retry")
    with socket.socket() as sock:
        # Permit the previous owned server's TIME_WAIT sockets, but still reject
        # a live listener. Match ThreadingHTTPServer's Linux reuse policy.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", a.port))
    runtime = ROOT / "build/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    lease = nullcontext()
    if a.mangai_root and not a.external_gpu_lease:
        sys.path.insert(0, str(a.mangai_root.resolve() / "manga-studio/python"))
        from mangai_gpu import GpuLease
        lease = GpuLease("coder", "GLM-5.3 Flash / custom Strata", root=a.mangai_root.resolve() / "work/gpu",
                         limits=(required_ram, a.min_vram_gib), timeout=60, cancelled=lambda: stopped,
                         progress=lambda text: print(text, flush=True))
    env = os.environ.copy()
    # Do not inherit experimental overrides from another Strata model/session.
    for key in list(env):
        if key.startswith("STRATA_GLM_"):
            del env[key]
    env.update(STRATA_GLM_SPLIT="0", STRATA_GLM_NO_MTP="1", STRATA_GLM_VRAM_GB=str(a.vram_cap_gib),
               STRATA_GLM_RESERVE_MB="2048", STRATA_GLM_RAM_HEADROOM_GB=str(a.ram_headroom_gib),
               STRATA_GLM_TIMING="1", STRATA_GLM_POOL_STATS="1",
               STRATA_GLM_PREFILL_MB=str(a.prefill_mb), STRATA_GLM_PREFILL_CHUNK=str(a.prefill_chunk),
               STRATA_GLM_PREFILL_MIN=str(a.prefill_min),
               STRATA_GLM_CPU_SPIN_US=str(a.cpu_spin_us),
               STRATA_GLM_RAM_EVICT=a.ram_policy,
               STRATA_GLM_RAM_PROTECT=str(a.ram_protect_tokens),
               STRATA_GLM_USAGE=str(pack / "expert_usage.txt"), OMP_NUM_THREADS="8")
    if a.cpu_lane == "off":
        env["STRATA_GLM_CPU_LANE"] = "0"
    elif a.cpu_threads is not None:
        env["STRATA_GLM_CPU_LANE"] = str(a.cpu_threads)
    if a.cpu_plan is not None:
        env["STRATA_GLM_CPU_PLAN"] = a.cpu_plan
    if a.expert_pool_gib is not None:
        env["STRATA_GLM_POOL_GB"] = str(a.expert_pool_gib)
    if a.ram_tier_gib is not None:
        env["STRATA_GLM_RAM_GB"] = str(a.ram_tier_gib)
    if a.uniform_expert_slots:
        env["STRATA_GLM_UNIFORM_SLOTS"] = "1"
    if a.usage_expert_slots:
        env["STRATA_GLM_USAGE_SLOTS"] = "1"
    if a.prefetch_experts:
        env["STRATA_GLM_PREFETCH_N"] = str(a.prefetch_experts)
    if a.lookahead_layers:
        env["STRATA_GLM_AHEAD"] = str(a.lookahead_layers)
    if a.benchmark:
        env["STRATA_GLM_NO_REUSE"] = "1"
    run_id = time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    if a.trace_experts:
        env["STRATA_GLM_ROUTE_LOG"] = str(runtime / f"{run_id}-routes")
    if a.profile_gpu_after is not None:
        env["STRATA_GLM_PROF"] = str(a.profile_gpu_after)
        env["STRATA_GLM_PROFILE_LOG"] = str(runtime / f"{run_id}-gpu-profile")
    usage_path = usage_seed
    if a.benchmark:
        usage_path = runtime / f"{run_id}-usage.txt"
        if usage_seed.is_file():
            shutil.copyfile(usage_seed, usage_path)
    env["STRATA_GLM_USAGE"] = str(usage_path)
    plan.update(usage_seed=str(usage_seed), usage_path=str(usage_path))
    config = {"exe": str(exe), "args": ["--glm-pack", str(pack), "--max-context", str(a.context)],
              "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"), "gpu": [0],
              "model_name": a.model_name, "host": "127.0.0.1",
              "log": str(runtime / f"{run_id}-engine.log"), "thinking_budget": a.thinking_budget}
    cfg_path = runtime / f"{run_id}.json"
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    (runtime / f"{run_id}-profile.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    metrics_path = runtime / f"{run_id}-resources.jsonl"
    child = None
    stopped = False

    def stop(_sig, _frame):
        nonlocal stopped
        stopped = True
        if child and child.poll() is None:
            try:
                if a.external_gpu_lease:
                    child.terminate()
                else:
                    os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    with open(runtime / "server.lock", "a") as lock, lease, ExitStack() as cleanup:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cleanup.enter_context(max_performance(a.prefer_max_performance))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        try:
            Path("/proc/self/oom_score_adj").write_text("500")
        except OSError:
            pass
        print(f"GPU・RAM負荷開始。context={a.context}。ログ: {runtime / run_id}", flush=True)
        command = [sys.executable, "-u", str(ROOT / "serve/server.py"), "--engine", "strata",
                   "--config", str(cfg_path), "--port", str(a.port)]
        if affinity:
            command = ["taskset", "--cpu-list", ",".join(map(str, sorted(affinity))), *command]
        # Under the broker all descendants stay in its owned group, so a launcher
        # crash cannot leave an engine outside the broker's cleanup boundary.
        child = subprocess.Popen(command, env=env, start_new_session=not a.external_gpu_lease)
        health_failures = 0
        slow_link_samples = 0
        slow_link_reported = False
        try:
            with open(metrics_path, "a", encoding="utf-8") as metrics:
                while child.poll() is None and not stopped:
                    info = resources()
                    ready = resident_ready(a.resident_health)
                    health_failures = 0 if ready else health_failures + 1
                    slow_link_samples = slow_link_samples + 1 if info["gpu_util_percent"] >= 50 and info["pcie_gen"] <= 2 else 0
                    if slow_link_samples >= 3 and not slow_link_reported:
                        print(f"注意: 高負荷中もPCIe Gen{info['pcie_gen']} x{info['pcie_width']}です。RAM転送が速度を制限する可能性があります。", flush=True)
                        slow_link_reported = True
                    metrics.write(json.dumps(dict(time=time.time(), resident_ready=ready, **info)) + "\n")
                    metrics.flush()
                    if info["ram_available_gib"] < 8 or info["vram_free_gib"] < 1 or health_failures >= 3:
                        raise RuntimeError("memory/A1 health guard stopped this GLM process")
                    for _ in range(10):
                        if stopped or child.poll() is not None:
                            break
                        time.sleep(1)
        finally:
            # Standalone owns the API group. Under the broker we share its group;
            # ignore our own TERM while asking all descendants to stop. The broker
            # performs the final group KILL after this launcher exits.
            if a.external_gpu_lease:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                os.killpg(os.getpgrp() if a.external_gpu_lease else child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            try:
                if a.external_gpu_lease:
                    if child.poll() is None:
                        child.kill()
                else:
                    os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        if child.returncode and not stopped:
            raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
