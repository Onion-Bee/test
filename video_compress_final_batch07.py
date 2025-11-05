#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch probe compressor with a STRONG structural-mode path for Laplacian variance in [100,250].
- Keeps your normal behavior for LV<100 and LV>250.
- For LV in [100,250] AND normal best S<0.50:
    Tier-A: x265 two-pass size-target @ C≈0.13 with tiny CAR + light denoise+unsharp
    Tier-B: SVT-AV1 CRF [52,50,48] on same CAR + filters (enable-qm=0)
    Tier-C: one-shot slightly stronger CAR (0.8×) with the winning encoder settings
- Evaluates with FAST VMAF (n_subsample=5).
- No outputs are saved; results printed and written to Excel (or CSV fallback).
"""

import os, sys, math, json, tempfile, shutil, subprocess, pathlib, statistics, time
from typing import List, Tuple, Dict, Optional

# ----------------------------
# Config / Paths
# ----------------------------
INPUT_DIR   = "video_5s"
EXCEL_PATH  = "video_5s_results_new.xlsx"
CSV_FALLBACK_PATH = "video_5s_results_new.csv"

FFMPEG_THREADS = "1"
VMAF_SUBSAMPLE = 5
VMAF_THREADS   = 1

# Primary codec by container (normal path)
PRIMARY_BY_EXT = {
    ".mp4": "av1",
    ".mkv": "av1",
    ".mov": "hevc",
    ".avi": "hevc",
}

# SVT-AV1 params
SVT_PARAMS_NORMAL   = "aq-mode=2:enable-qm=1:enable-cdef=1:enable-restoration=1:scd=1"
SVT_PARAMS_FLATSAFE = "aq-mode=2:enable-qm=0:enable-cdef=1:enable-restoration=1:scd=1"  # for flat content

# x265 params
X265_PARAMS_NORMAL   = "aq-mode=3:rd=4:limit-sao=1:rect=1:amp=1"
# Structural mode: SSIM-oriented (reduced psy), better for metrics on flat content
X265_PARAMS_SSIM     = "aq-mode=2:aq-strength=0.8:rd=4:limit-sao=1:rect=1:amp=1:tune=ssim"

# ----------------------------
# Small utils
# ----------------------------
def run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def ffprobe_json(path: str) -> dict:
    code, out, err = run(["ffprobe","-v","error","-print_format","json","-show_format","-show_streams", path])
    if code != 0:
        raise RuntimeError(f"ffprobe failed: {err}")
    return json.loads(out)

def duration_sec(path: str) -> float:
    return float(ffprobe_json(path)["format"].get("duration", 0.0))

def file_size(path: str) -> int:
    return os.path.getsize(path)

def color_flags() -> List[str]:
    return ["-color_primaries","bt709","-color_trc","bt709","-colorspace","bt709","-color_range","tv"]

def list_videos(folder: str) -> List[str]:
    vids = []
    for n in sorted(os.listdir(folder)):
        p = os.path.join(folder, n)
        if os.path.isfile(p) and pathlib.Path(p).suffix.lower() in (".mp4",".mkv",".mov",".avi"):
            vids.append(p)
    return vids

# ----------------------------
# Laplacian variance (OpenCV)
# ----------------------------
def compute_laplacian_variance(path: str, sample_frames: int = 20) -> Optional[float]:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0: cap.release(); return None
    idxs = [max(0, int(i*(total-1)/(sample_frames-1))) for i in range(sample_frames)]
    vals = []
    for frame_idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vals.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    cap.release()
    return statistics.median(vals) if vals else None

# ----------------------------
# FAST VMAF
# ----------------------------
def build_vmaf_filter(subsample: int, threads: int, log_path: str) -> str:
    opts = [f"n_threads={threads}", "log_fmt=json", f"log_path='{log_path}'"]
    if subsample and subsample > 1:
        opts.append(f"n_subsample={subsample}")
    ref_chain  = "setpts=PTS-STARTPTS,format=yuv420p,setsar=1"
    dist_chain = "setpts=PTS-STARTPTS,format=yuv420p,setsar=1"
    return f"[0:v]{ref_chain}[ref];[1:v]{dist_chain}[dist];[dist][ref]libvmaf=" + ":".join(opts)

def compute_vmaf_fast(ref_path: str, dist_path: str) -> float:
    with tempfile.TemporaryDirectory() as td:
        logp = os.path.join(td, "vmaf.json")
        vfgraph = build_vmaf_filter(VMAF_SUBSAMPLE, VMAF_THREADS, logp)
        cmd = ["ffmpeg","-hide_banner","-loglevel","error","-nostdin",
               "-i", ref_path, "-i", dist_path, "-lavfi", vfgraph, "-f","null","-"]
        code, _, err = run(cmd)
        if code != 0:
            raise RuntimeError(f"fast VMAF failed:\n{err}")
        data = json.load(open(logp))
        try:
            return float(data["pooled_metrics"]["vmaf"]["mean"])
        except:
            vals = [f["metrics"]["vmaf"] for f in data.get("frames",[]) if "metrics" in f and "vmaf" in f["metrics"]]
            if not vals:
                raise RuntimeError("No VMAF frames")
            return sum(vals)/len(vals)

# ----------------------------
# S metric (your definition)
# ----------------------------
def s_metric(C, V, vmaf_threshold=80, compression_weight=0.7, quality_weight=0.3, soft_threshold_margin=5.0):
    hard_cutoff = vmaf_threshold - soft_threshold_margin
    if V < hard_cutoff:
        return 0.0
    if V < vmaf_threshold:
        soft_pos = (V - hard_cutoff) / soft_threshold_margin
        quality_factor = 0.7 * (soft_pos ** 2)
        if C >= 0.95:
            compression_component = 0.0
        else:
            ratio = 1.0 / C
            compression_component = ((ratio - 1) / 19) ** 1.5 if ratio <= 20 else 1.0 + 0.3 * math.log(ratio / 20.0)
            compression_component = min(1.3, compression_component)
        return min(1.0, compression_component * quality_factor)
    vmaf_excess = V - vmaf_threshold
    quality_component = 0.7 + 0.3 * min(1.0, vmaf_excess / (100.0 - vmaf_threshold))
    if C >= 0.95: compression_component = 0.0
    elif C >= 0.80: compression_component = (1.0 / C - 1.0) ** 2 * 0.4
    else:
        ratio = 1.0 / C
        compression_component = ((ratio - 1.25) / 18.75) ** 1.2 + 0.025 if ratio <= 20 else 1.0 + 0.3 * math.log(ratio / 20.0)
        compression_component = min(1.3, compression_component)
    return min(1.0, compression_weight * compression_component + quality_weight * quality_component)

# ----------------------------
# Strategy selection (normal)
# ----------------------------
def choose_strategy(meta: dict, lap_var: Optional[float]) -> Dict:
    ext = meta["ext"]; W = meta["W"]; H = meta["H"]
    native_is_1080p_or_more = (W >= 1900 or H >= 1000)
    native_is_720p_or_less  = (W <= 1300 or H <= 800)

    LV_LOW  = 120
    LV_MID  = 350

    work_res = None
    bucket   = "A_mid"
    DS = False
    if native_is_720p_or_less: DS = True
    elif native_is_1080p_or_more and lap_var is not None and lap_var < LV_LOW: DS = True

    if not DS:
        if lap_var is not None and lap_var > LV_MID:
            bucket = "A_high"; work_res = None
            crf_ladder = [44, 42, 40, 38, 36, 34, 32] if PRIMARY_BY_EXT.get(ext,"av1")=="av1" else [30, 28, 26]
        else:
            bucket = "A_mid"
            crf_ladder = [52, 50, 48, 46, 44, 42, 40, 38, 36] if PRIMARY_BY_EXT.get(ext,"av1")=="av1" else [32, 30, 28]
            if native_is_1080p_or_more and (lap_var is not None and lap_var < 200):
                work_res = (1280, 720)
    else:
        if lap_var is not None and lap_var < LV_LOW:
            bucket = "B_low"
            crf_ladder = [52, 50, 48, 46, 44] if PRIMARY_BY_EXT.get(ext,"av1")=="av1" else [34, 32, 30]
        else:
            bucket = "B_mid"
            crf_ladder = [52, 50, 48, 46, 44, 42, 40] if PRIMARY_BY_EXT.get(ext,"av1")=="av1" else [32, 30, 28]

    return {"bucket": bucket, "work_res": work_res, "primary": PRIMARY_BY_EXT.get(ext,"av1"),
            "crf_ladder": crf_ladder, "DS_flag": DS, "LV_bins": {"LV_LOW": LV_LOW, "LV_MID": LV_MID}}

# ----------------------------
# Encode trial (CRF) with optional overrides and filters
# ----------------------------
def encode_trial(src: str, meta: dict, crf: int, work_res: Optional[Tuple[int,int]],
                 codec_override: Optional[str] = None,
                 vf_filters: Optional[List[str]] = None,
                 svt_params: Optional[str] = None,
                 x265_params: Optional[str] = None) -> Tuple[bool, str, str, str]:
    ext = meta["ext"]
    codec = codec_override or PRIMARY_BY_EXT.get(ext, "av1")
    W, H = meta["W"], meta["H"]
    w_out, h_out = (work_res if work_res else (W, H))

    tmpdir = tempfile.mkdtemp(prefix="enc_")
    out_ext = ".mp4" if ext in (".mp4", ".mov") else ".mkv"
    outp   = os.path.join(tmpdir, f"enc{out_ext}")

    vf_chain = []
    if (w_out, h_out) != (W, H):
        vf_chain.append(f"scale={w_out}:{h_out}:flags=lanczos+accurate_rnd")
    if vf_filters:
        vf_chain.extend(vf_filters)
    vf_chain.append("format=yuv420p10le")
    vf = ",".join(vf_chain) if vf_chain else "format=yuv420p10le"

    if codec == "av1":
        params = svt_params or SVT_PARAMS_NORMAL
        cmd = ["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-threads",FFMPEG_THREADS,
               "-y","-i",src,"-vf",vf,*color_flags(),
               "-c:v","libsvtav1","-crf",str(crf),"-preset","6","-pix_fmt","yuv420p10le",
               "-svtav1-params",params,"-an","-movflags","+faststart", outp]
    else:
        params = x265_params or X265_PARAMS_NORMAL
        tag = ["-tag:v","hvc1"] if ext in (".mp4",".mov") else []
        cmd = ["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-threads",FFMPEG_THREADS,
               "-y","-i",src,"-vf",vf,*color_flags(), *tag,
               "-c:v","libx265","-crf",str(crf),"-preset","medium","-pix_fmt","yuv420p10le",
               "-x265-params",params,"-an", outp]

    code,_,err = run(cmd)
    if code!=0 or not os.path.exists(outp) or file_size(outp)==0:
        shutil.rmtree(tmpdir, True)
        return False, "", "", f"encode failed: {err}"
    return True, outp, tmpdir, codec

# ----------------------------
# Two-pass bitrate-targeted x265 (size-target)
# ----------------------------
def encode_x265_2pass_bitrate(src: str, meta: dict, bitrate_kbps: int, work_res: Optional[Tuple[int,int]],
                              vf_filters: Optional[List[str]] = None) -> Tuple[bool, str, str, str]:
    ext = meta["ext"]
    W, H = meta["W"], meta["H"]
    w_out, h_out = (work_res if work_res else (W, H))

    tmpdir = tempfile.mkdtemp(prefix="enc2p_")
    out_ext = ".mp4" if ext in (".mp4",".mov") else ".mkv"
    outp   = os.path.join(tmpdir, f"enc{out_ext}")

    vf_chain = []
    if (w_out, h_out) != (W, H):
        vf_chain.append(f"scale={w_out}:{h_out}:flags=lanczos+accurate_rnd")
    if vf_filters:
        vf_chain.extend(vf_filters)
    vf_chain.append("format=yuv420p10le")
    vf = ",".join(vf_chain)

    tag = ["-tag:v","hvc1"] if ext in (".mp4",".mov") else []
    stats = os.path.join(tmpdir, "x265_2pass.log")

    # Pass 1
    cmd1 = ["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-threads",FFMPEG_THREADS,
            "-y","-i",src,"-vf",vf,*color_flags(), *tag,
            "-c:v","libx265","-b:v",f"{bitrate_kbps}k","-preset","medium","-pix_fmt","yuv420p10le",
            "-x265-params",f"{X265_PARAMS_SSIM}:pass=1:stats={stats}",
            "-an","-f","mp4","/dev/null"]
    code1,_,err1 = run(cmd1)
    if code1 != 0:
        shutil.rmtree(tmpdir, True); return False, "", "", f"2pass pass1 failed: {err1}"

    # Pass 2
    cmd2 = ["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-threads",FFMPEG_THREADS,
            "-y","-i",src,"-vf",vf,*color_flags(), *tag,
            "-c:v","libx265","-b:v",f"{bitrate_kbps}k","-preset","medium","-pix_fmt","yuv420p10le",
            "-x265-params",f"{X265_PARAMS_SSIM}:pass=2:stats={stats}",
            "-an", outp]
    code2,_,err2 = run(cmd2)
    if code2 != 0 or not os.path.exists(outp) or file_size(outp)==0:
        shutil.rmtree(tmpdir, True); return False, "", "", f"2pass pass2 failed: {err2}"

    # clean stats (ffmpeg may write .mbtree, etc.)
    for n in os.listdir(tmpdir):
        if n.endswith(".log") or n.endswith(".mbtree"):
            try: os.remove(os.path.join(tmpdir,n))
            except: pass

    return True, outp, tmpdir, "hevc-2pass"

# ----------------------------
# Build "view-like" for VMAF
# ----------------------------
def build_view_like_for_vmaf(encoded_path: str, meta: dict) -> Tuple[bool,str,str]:
    W, H = meta["W"], meta["H"]
    tmpdir = tempfile.mkdtemp(prefix="vlike_")
    outp = os.path.join(tmpdir, "view_like.mp4")
    vf = f"scale={W}:{H}:flags=spline+accurate_rnd,format=yuv420p,setsar=1"
    cmd = ["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-threads",FFMPEG_THREADS,
           "-y","-i",encoded_path,"-vf",vf,*color_flags(),
           "-c:v","libx264","-crf","18","-preset","fast","-pix_fmt","yuv420p","-an", outp]
    code,_,err = run(cmd)
    if code!=0 or not os.path.exists(outp) or file_size(outp)==0:
        shutil.rmtree(tmpdir, True)
        return False, "", f"reconstruct failed: {err}"
    return True, outp, tmpdir

# ----------------------------
# CAR helpers for the structural band
# ----------------------------
def car_tiny_dims(W,H):
    # ~0.83× per dimension for 1080p; ~0.90× for 720p
    if W >= 1900 or H >= 1000:
        return (1600, 900)
    elif W >= 1200 or H >= 700:
        return (int(round(W*0.90)), int(round(H*0.90)))
    else:
        return (W, H)

def car_strong_dims(W,H):
    # one notch stronger ~0.8×
    return (int(round(W*0.80)), int(round(H*0.80)))

def prefilter_chain():
    # very light denoise + mild unsharp to hold edges
    # (hqdn3d is broadly available and safe; unsharp modest)
    return ["hqdn3d=1.0:1.0:4:4", "unsharp=3:3:0.5:3:3:0.0"]

# ----------------------------
# Structural-mode pipeline (FOR 100<=LV<=250 ONLY)
# ----------------------------
def structural_mode_pipeline(src: str, meta: dict, best_so_far: Optional[Dict]) -> Optional[Dict]:
    """
    Tier-A: x265 2-pass size target @ C≈0.13 on tiny CAR + prefilter (2-3 tries with +-10%)
    Tier-B: AV1 CRF [52,50,48] on same CAR + prefilter
    Tier-C: one-shot stronger CAR (0.8×) with winner encoder settings
    Returns improved dict or None.
    """
    size_bytes = file_size(src)
    dur = duration_sec(src)
    W, H = meta["W"], meta["H"]

    work_res_tiny   = car_tiny_dims(W,H)
    work_res_strong = car_strong_dims(W,H)
    vf_filters      = prefilter_chain()

    improved = best_so_far

    # --- Tier-A: x265 2-pass at target C ≈ 0.13 (≈7.7× smaller) ---
    C_target = 0.13
    target_bps = int((size_bytes * C_target * 8) / max(0.001, dur))   # bits per second
    target_kbps = max(50, target_bps // 1000)

    for mult in (1.0, 0.9, 1.1):   # small sweep around target
        b_kbps = int(target_kbps * mult)
        ok, enc, td, used = encode_x265_2pass_bitrate(src, meta, b_kbps, work_res_tiny, vf_filters=vf_filters)
        if not ok: continue
        ok2, vlike, td2 = build_view_like_for_vmaf(enc, meta)
        if not ok2:
            shutil.rmtree(td, True); continue
        try:
            V = compute_vmaf_fast(src, vlike)
        except Exception:
            V = 0.0
        C = file_size(enc)/size_bytes
        S = s_metric(C, V)
        cur = {"file": os.path.basename(src), "duration_sec": dur,
               "VMAF": V, "C": C, "S": S, "CRF": f"2pass@{b_kbps}k",
               "If_Downscaled": f"{work_res_tiny[0]}x{work_res_tiny[1]}",
               "note": f"struct-hevc-2pass-{b_kbps}k"}
        if (improved is None) or (S > improved["S"]):
            improved = cur
        shutil.rmtree(td2, True); shutil.rmtree(td, True)
        if S >= 0.50:
            return improved

    # --- Tier-B: AV1 CRF mini sweep on same CAR + filters ---
    for crf in (52, 50, 48):
        ok, enc, td, used = encode_trial(src, meta, crf, work_res_tiny,
                                         codec_override="av1", vf_filters=vf_filters,
                                         svt_params=SVT_PARAMS_FLATSAFE)
        if not ok: continue
        ok2, vlike, td2 = build_view_like_for_vmaf(enc, meta)
        if not ok2:
            shutil.rmtree(td, True); continue
        try:
            V = compute_vmaf_fast(src, vlike)
        except Exception:
            V = 0.0
        C = file_size(enc)/size_bytes
        S = s_metric(C, V)
        cur = {"file": os.path.basename(src), "duration_sec": dur,
               "VMAF": V, "C": C, "S": S, "CRF": crf,
               "If_Downscaled": f"{work_res_tiny[0]}x{work_res_tiny[1]}",
               "note": f"struct-av1-crf{crf}"}
        if (improved is None) or (S > improved["S"]):
            improved = cur
        shutil.rmtree(td2, True); shutil.rmtree(td, True)
        if S >= 0.50:
            return improved

    # --- Tier-C: one-shot stronger CAR @ 0.8× using winner settings ---
    if improved:
        # detect winner codec flavor
        use_av1 = ("struct-av1" in improved.get("note",""))
        if "struct-hevc-2pass" in improved.get("note",""): use_av1 = False

        if use_av1:
            # re-run AV1 once at stronger CAR with same CRF (if numeric) or 50
            crf = improved["CRF"] if isinstance(improved["CRF"], int) else 50
            ok, enc, td, _ = encode_trial(src, meta, crf, work_res_strong,
                                          codec_override="av1", vf_filters=vf_filters,
                                          svt_params=SVT_PARAMS_FLATSAFE)
        else:
            # re-run 2-pass x265 with same bitrate @ stronger CAR
            # parse kbps from note: "struct-hevc-2pass-XXXXk"
            try:
                kbps = int(str(improved.get("note","")).split("2pass-")[1].split("k")[0])
            except Exception:
                kbps = int((size_bytes * 0.13 * 8) / max(0.001, dur) / 1000)
            ok, enc, td, _ = encode_x265_2pass_bitrate(src, meta, kbps, work_res_strong, vf_filters=vf_filters)

        if ok:
            ok2, vlike, td2 = build_view_like_for_vmaf(enc, meta)
            if ok2:
                try:
                    V = compute_vmaf_fast(src, vlike)
                except Exception:
                    V = 0.0
                C = file_size(enc)/size_bytes
                S = s_metric(C, V)
                cur = {"file": os.path.basename(src), "duration_sec": dur,
                    "VMAF": V, "C": C, "S": S, "CRF": improved["CRF"],
                    "If_Downscaled": f"{work_res_strong[0]}x{work_res_strong[1]}",
                    "note": ("structC-av1" if use_av1 else "structC-hevc")}
                if S > improved["S"]:
                    improved = cur
            if ok2: shutil.rmtree(td2, True)
            shutil.rmtree(td, True)

    return improved if (best_so_far and improved and improved["S"] > best_so_far["S"]) else None

# ----------------------------
# Normal path (unchanged)
# ----------------------------
def process_normal(src: str, meta: dict, strat: dict) -> Optional[Dict]:
    size_bytes = file_size(src)
    dur = duration_sec(src)
    work_res = strat["work_res"]
    crf_ladder = strat["crf_ladder"]
    downscaled_tag = f"{work_res[0]}x{work_res[1]}" if work_res else "NO"
    best, prev_S = None, None

    for crf in crf_ladder:
        ok, enc, td, used_codec = encode_trial(src, meta, crf, work_res)
        if not ok: continue
        ok2, vlike, td2 = build_view_like_for_vmaf(enc, meta)
        if not ok2:
            shutil.rmtree(td, True); continue
        try:
            V = compute_vmaf_fast(src, vlike)
        except Exception:
            V = 0.0
        C = file_size(enc)/size_bytes
        S = s_metric(C, V)
        cur = {"file": os.path.basename(src), "duration_sec": dur,
               "VMAF": V, "C": C, "S": S, "CRF": crf,
               "If_Downscaled": downscaled_tag, "note": f"normal-{used_codec}"}
        if (best is None) or (S > best["S"]):
            best = cur

        # Early stops
        if S > 0.70 or ((V > 87.0) and (C < 0.11)):
            shutil.rmtree(td2, True); shutil.rmtree(td, True)
            break
        # Guard cliff
        if (prev_S is not None) and (V < 80.0) and (S < prev_S):
            shutil.rmtree(td2, True); shutil.rmtree(td, True)
            break
        prev_S = S
        shutil.rmtree(td2, True); shutil.rmtree(td, True)

    return best

# ----------------------------
# One video wrapper
# ----------------------------
def process_one(src: str) -> Dict:
    meta_j = ffprobe_json(src)
    v0 = [s for s in meta_j["streams"] if s.get("codec_type")=="video"][0]
    meta = {"W": int(v0["width"]), "H": int(v0["height"]), "fps": v0.get("avg_frame_rate","24/1"),
            "ext": pathlib.Path(src).suffix.lower()}
    dur = float(meta_j["format"].get("duration", 0.0))

    lap_var = compute_laplacian_variance(src, sample_frames=20)
    strat = choose_strategy(meta, lap_var)

    print(f"=== {os.path.basename(src)} ===")
    print(f"Laplace (variance): {('NA' if lap_var is None else f'{lap_var:.1f}')}")
    print(f"Strategy: bucket={strat['bucket']}, DS_flag={strat['DS_flag']}, "
          f"work_res={('NO' if strat['work_res'] is None else f'{strat['work_res'][0]}x{strat['work_res'][1]}')}, "
          f"primary={strat['primary']}, CRFs={strat['crf_ladder']}")

    # Normal path first
    best = process_normal(src, meta, strat)

    # Structural-mode only for 100<=LV<=250 and best S<0.50
    if (lap_var is not None) and (100.0 <= lap_var <= 250.0) and (best is not None) and (best["S"] < 0.50):
        improved = structural_mode_pipeline(src, meta, best)
        if improved and improved["S"] > best["S"]:
            best = improved

    if best is None:
        return {"file": os.path.basename(src), "duration_sec": dur,
                "VMAF": None, "C": None, "S": 0.0, "CRF": None,
                "If_Downscaled": "NO", "lap_var": lap_var, "note": "no-encode"}

    best["lap_var"] = lap_var
    return best

# ----------------------------
# Excel writer (CSV fallback)
# ----------------------------
def write_table(rows: List[Dict], xlsx_path: str, csv_fallback: str):
    table = []
    for r in rows:
        table.append({
            "Video Name": r.get("file",""),
            "Laplace Variance": r.get("lap_var",""),
            "CRF":        r.get("CRF", None),
            "VMAF":       None if r.get("VMAF") is None else float(r["VMAF"]),
            "Compression":None if r.get("C")   is None else float(r["C"]),
            "S value":    None if r.get("S")   is None else float(r["S"]),
        })
    try:
        import pandas as pd
        df = pd.DataFrame(table, columns=["Video Name", "Laplace Variance", "CRF","VMAF","Compression","S value"])
        df.to_excel(xlsx_path, index=False)
        print(f"\nExcel written to: {xlsx_path}")
    except Exception as e:
        try:
            import csv
            with open(csv_fallback, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["Video Name","CRF","VMAF","Compression","S value"])
                writer.writeheader()
                writer.writerows(table)
            print(f"\n(openpyxl/pandas not available) CSV written to: {csv_fallback}")
        except Exception as e2:
            print(f"\n[WARN] Could not write Excel or CSV: {e} / {e2}")

# ----------------------------
# Main
# ----------------------------
def main():
    videos = list_videos(INPUT_DIR)
    if not videos:
        print(f"No videos in {INPUT_DIR}"); return

    start = time.time()
    all_results = []
    for v in videos:
        try:
            res = process_one(v)
            all_results.append(res)
            vmaf_str = "NA" if (res.get("VMAF") is None) else f"{res['VMAF']:.2f}"
            c_str    = "NA" if (res.get("C")    is None) else f"{res['C']:.4f}"
            s_str    = "NA" if (res.get("S")    is None) else f"{res['S']:.3f}"
            crf_str  = str(res.get("CRF","NA"))
            print(f"{res['file']} | VMAF={vmaf_str} | C={c_str} | S={s_str} | CRF={crf_str} | If_Downscaled={res.get('If_Downscaled','NO')} | {res.get('note','')}")
        except Exception as e:
            print(f"[ERROR] {v}: {e}")

    print("\nJSON Results:")
    print(json.dumps(all_results, indent=2))
    write_table(all_results, EXCEL_PATH, CSV_FALLBACK_PATH)
    end = time.time()
    print("Elapsed Time = ", (end - start))

if __name__ == "__main__":
    main()
