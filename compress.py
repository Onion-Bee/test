#!/usr/bin/env python3
"""
Modified H.265/HEVC compression script for MP4-only with 95+ VMAF and 10x compression targets
Modifications:
  - MP4 container only, libx265 encoder only
  - CRF sweep from low to high until both VMAF ≥95 and compression ≥10x are achieved
  - Veryslow preset for maximum quality
  - Time doesn't matter, quality is priority
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import math
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction

sys.path.append(str(Path(__file__).parent))
try:
    from score import calculate_compression_score
except Exception:
    def calculate_compression_score(**kwargs):
        vmaf_score = kwargs.get('vmaf_score', 0)
        compression_rate = kwargs.get('compression_rate', 1.0)
        return vmaf_score - compression_rate * 10, compression_rate, vmaf_score, 'fallback'


# ---- Defaults for the fast VMAF function ----
SAMPLING_VMAF_SUBSAMPLE = 8
SAMPLING_VMAF_DOWNSCALE_HALF = False
VMAF_THREADS = 0


def run_cmd(args: List[str], timeout: int = 3600) -> Tuple[int, str, str]:
    """Run command with extended timeout for quality encoding"""
    if not args:
        raise ValueError("run_cmd requires a non-empty args list")
    cmd = args[:] if args[0].lower().endswith("ffmpeg") or os.path.basename(args[0]).lower() == "ffmpeg" else ["ffmpeg"] + args
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False
        )
        return proc.returncode, (proc.stdout or ""), (proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        return -1, "", f"TimeoutExpired: {e}"


def _probe_video_props(path: str) -> Tuple[int, int, float]:
    """Returns (width, height, fps) for the first video stream in the file."""
    try:
        res = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', path
        ], capture_output=True, text=True, timeout=30)
        if res.returncode != 0 or not res.stdout:
            return 0, 0, 0.0
        info = json.loads(res.stdout)
        for s in info.get('streams', []):
            if s.get('codec_type') == 'video':
                w = int(s.get('width') or 0)
                h = int(s.get('height') or 0)
                rfr = s.get('r_frame_rate') or s.get('avg_frame_rate') or '0/1'
                try:
                    fps = float(Fraction(rfr))
                except Exception:
                    fps = 0.0
                return w, h, fps
    except Exception:
        pass
    return 0, 0, 0.0


def vmaf_mean_aligned_fast(ref: str,
                           dist: str,
                           src_for_norm: Optional[str] = None,
                           n_subsample: int = SAMPLING_VMAF_SUBSAMPLE,
                           half_res: bool = SAMPLING_VMAF_DOWNSCALE_HALF,
                           vmaf_threads: int = VMAF_THREADS) -> float:
    """Fast sampling VMAF calculation"""
    srcn = src_for_norm or ref
    w, h, fps = _probe_video_props(srcn)
    scale_filter = ""
    fps_filter = ""
    if w > 0 and h > 0:
        if half_res:
            w = max(1, w // 2)
            h = max(1, h // 2)
        scale_filter = f"scale={w}:{h}:flags=bicubic,"
    if fps > 0:
        fps_filter = f"fps=fps={fps},"

    with tempfile.TemporaryDirectory() as td:
        logp = os.path.join(td, "vmaf.json")
        opts = [f"n_threads={vmaf_threads}", "log_fmt=json", f"log_path={logp}"]
        if n_subsample and n_subsample > 1:
            opts.append(f"n_subsample={n_subsample}")

        lavfi = (
            f"[0:v]setpts=PTS-STARTPTS,{scale_filter}{fps_filter}format=yuv420p[ref];"
            f"[1:v]setpts=PTS-STARTPTS,{scale_filter}{fps_filter}format=yuv420p[dist];"
            f"[ref][dist]libvmaf=" + ":".join(opts)
        )

        args = [
            "-hide_banner",
            "-y",
            "-i", str(Path(ref).resolve()),
            "-i", str(Path(dist).resolve()),
            "-lavfi", lavfi,
            "-f", "null", "-"
        ]
        code, out, err = run_cmd(args, timeout=3600)
        if code != 0:
            raise RuntimeError(f"VMAF (fast) ffmpeg failed (code={code}): {err[-2000:]}")

        if not os.path.exists(logp):
            raise RuntimeError("VMAF JSON log not found after libvmaf run")
        try:
            with open(logp, "r") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read/parse vmaf JSON log: {e}")

        try:
            return float(data["pooled_metrics"]["vmaf"]["mean"])
        except Exception:
            frames = data.get("frames", [])
            vals = [fr.get("metrics", {}).get("vmaf") for fr in frames if "metrics" in fr and "vmaf" in fr.get("metrics", {})]
            vals = [float(v) for v in vals if v is not None]
            if not vals:
                raise RuntimeError("VMAF JSON missing values")
            return sum(vals) / len(vals)


class H265Compressor:
    def __init__(self, config_file: str = "config.json"):
        self.config = self._load_config(config_file)
        self.start_time = None

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        config_path = Path(__file__).parent / config_file
        default_config = {
            "algorithm_name": "H.265 High Efficiency",
            "parameters": {
                "preset": "veryslow",
                "crf": 18,
                "profile": "main",
                "level": "4.1",
                "tune": "none",
                "threads": 0,
                "tile_columns": 2,
                "tile_rows": 1
            },
            "audio": {
                "codec": "aac",
                "bitrate": "128k",
                "sample_rate": 44100
            },
            "optimization_params": {
                "crf_min": 14,
                "crf_max": 35,
                "vmaf_threshold": 95,
                "compression_ratio_target": 10
            }
        }
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    loaded = json.load(f)
                default_config.update(loaded)
            except Exception as e:
                print(f"Warning: failed to load config {config_file}: {e}")
        return default_config

    def _validate_input(self, input_video: str) -> bool:
        if not os.path.exists(input_video):
            print(f"Error: Input video file '{input_video}' does not exist!")
            return False
        if not input_video.lower().endswith('.mp4'):
            print(f"Error: Only MP4 files are supported. Got: {input_video}")
            return False
        try:
            with open(input_video, 'rb') as f:
                f.read(1024)
        except IOError:
            print(f"Error: Cannot read input video file '{input_video}'!")
            return False
        return True

    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ], capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout:
                return json.loads(result.stdout)
        except Exception:
            pass
        return {}

    def _build_x265_params(self, params: Dict[str, Any]) -> str:
        """Build optimized x265 parameters for maximum quality"""
        x265_params = []
        tile_cols = params.get('tile_columns', 2)
        tile_rows = params.get('tile_rows', 1)
        if tile_cols > 0 and tile_rows > 0:
            x265_params.append(f"tiles={tile_cols}x{tile_rows}")
        
        # High quality parameters
        x265_params.extend([
            "rc-lookahead=60",
            "bframes=16",
            "b-adapt=2",
            "ref=6",
            "me=star",
            "subme=7",
            "rd=6",
            "psy-rd=2.0",
            "psy-rdoq=1.0",
            "aq-mode=3",
            "aq-strength=1.0",
            "deblock=-1:-1",
            "sao=1",
            "limit-sao=1",
            "rect=1",
            "amp=1",
            "max-merge=5",
            "temporal-mvp=1",
            "weightp=1",
            "weightb=1",
            "analyze-src-pics=1",
            "hrd=1"
        ])
        return ":".join(x265_params)

    def _build_ffmpeg_command(self, input_video: str, output_video: str) -> List[str]:
        params = self.config.get('parameters', {})
        audio_params = self.config.get('audio', {})

        # Force MP4 output and libx265 encoder
        video_encode_flags = [
            '-c:v', 'libx265',
            '-preset', str(params.get('preset', 'veryslow')),
            '-crf', str(params.get('crf', 18)),
            '-profile:v', params.get('profile', 'main'),
            '-level', params.get('level', '4.1'),
            '-x265-params', self._build_x265_params(params),
            '-pix_fmt', 'yuv420p'
        ]

        # Audio settings
        audio_flags = [
            '-c:a', audio_params.get('codec', 'aac'),
            '-b:a', audio_params.get('bitrate', '128k'),
            '-ar', str(audio_params.get('sample_rate', 44100))
        ]

        # Probe flags for better analysis
        probe_flags = ['-probesize', '100M', '-analyzeduration', '100M']

        cmd = [
            'ffmpeg', '-y'
        ] + probe_flags + [
            '-i', input_video
        ] + video_encode_flags + audio_flags + [
            output_video
        ]

        return cmd

    def _compute_vmaf(self, ref_video: str, dist_video: str) -> float:
        """Primary VMAF computation with fallback"""
        print(f"[VMAF] Starting VMAF calculation...")
        try:
            score = vmaf_mean_aligned_fast(ref_video, dist_video)
            print(f"[VMAF] Fast VMAF result: {score:.3f}")
            return score
        except Exception as e:
            print(f"[VMAF] Fast VMAF failed: {e}. Falling back to robust method...")
            return self._compute_vmaf_fallback(ref_video, dist_video)

    def _compute_vmaf_fallback(self, ref_video: str, dist_video: str) -> float:
        """Robust JSON-based ffmpeg/libvmaf pipeline"""
        print(f"[VMAF] Starting VMAF fallback calculation...")
        try:
            ref_info = self._get_video_info(ref_video)
            ref_stream = self._get_primary_video_stream(ref_info)
            if not ref_stream:
                print("[VMAF] Could not read reference stream info; skipping VMAF")
                return 0.0

            w = int(ref_stream.get('width', 0) or 0)
            h = int(ref_stream.get('height', 0) or 0)
            rfr = ref_stream.get('r_frame_rate') or ref_stream.get('avg_frame_rate') or '0/1'
            try:
                fps = float(Fraction(rfr))
            except Exception:
                fps = 0.0

            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as vmaf_log:
                log_path = vmaf_log.name

            scale_filter = ''
            if w > 0 and h > 0:
                scale_filter = f"scale={w}:{h}:flags=bicubic,"

            fps_filter = ''
            if fps > 0:
                fps_filter = f"fps=fps={fps},"

            lavfi = (
                f"[0:v]setpts=PTS-STARTPTS,{scale_filter}{fps_filter}format=yuv420p[ref];"
                f"[1:v]setpts=PTS-STARTPTS,{scale_filter}{fps_filter}format=yuv420p[dist];"
                f"[ref][dist]libvmaf=log_fmt=json:log_path={log_path}"
            )

            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i", str(Path(ref_video).resolve()),
                "-i", str(Path(dist_video).resolve()),
                "-lavfi", lavfi,
                "-f", "null", "-"
            ]

            print(f"[VMAF] Running fallback ffmpeg/libvmaf...")
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3600,
                check=False
            )

            if Path(log_path).exists() and Path(log_path).stat().st_size > 10:
                try:
                    with open(log_path, "r") as f:
                        vmaf_data = json.load(f)
                    frames = vmaf_data.get('frames', [])
                    if frames:
                        vmaf_scores = [float(frame.get('metrics', {}).get('vmaf', 0)) for frame in frames if 'vmaf' in frame.get('metrics', {})]
                        if vmaf_scores:
                            score = sum(vmaf_scores) / len(vmaf_scores)
                            print(f"[VMAF] Parsed JSON VMAF (fallback): {score:.3f}")
                            return score
                except Exception as e:
                    print(f"[VMAF] JSON parse error in fallback: {e}")

            print(f"[VMAF] VMAF fallback calculation failed.")
            return 0.0
        finally:
            try:
                if 'log_path' in locals() and Path(log_path).exists():
                    Path(log_path).unlink()
            except Exception:
                pass

    def _get_primary_video_stream(self, info: Dict[str, Any]) -> Dict[str, Any]:
        for s in info.get('streams', []):
            if s.get('codec_type') == 'video':
                return s
        return {}

    def _encode_with_crf(self, input_video: str, crf: int) -> Tuple[Optional[str], float, float]:
        """Encode with specific CRF and return (output_file, vmaf_score, compression_ratio)"""
        params = dict(self.config.get("parameters", {}))
        params["crf"] = crf

        tmp_output = Path(tempfile.gettempdir()) / f"tmp_crf{crf}.mp4"
        if tmp_output.exists():
            try:
                tmp_output.unlink()
            except Exception:
                pass

        saved_config = dict(self.config)
        try:
            self.config = dict(self.config)
            self.config['parameters'] = params

            cmd = self._build_ffmpeg_command(input_video, str(tmp_output))
            print(f"[ENCODE] Encoding with CRF={crf}...")
            
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
            if proc.returncode != 0:
                print(f"[ENCODE] ffmpeg failed for CRF={crf} rc={proc.returncode}")
                print((proc.stderr or '')[-1000:])
                return None, 0.0, 0.0

            if not tmp_output.exists():
                return None, 0.0, 0.0

            orig_size = os.path.getsize(input_video)
            comp_size = os.path.getsize(tmp_output)
            compression_ratio = orig_size / comp_size if comp_size > 0 else 0.0

            vmaf_score = self._compute_vmaf(input_video, str(tmp_output))

            print(f"[RESULT] CRF={crf}: VMAF={vmaf_score:.2f}, Compression={compression_ratio:.2f}x, Size={comp_size/(1024*1024):.1f}MB")
            
            return str(tmp_output), vmaf_score, compression_ratio

        except subprocess.TimeoutExpired:
            print(f"[ENCODE] CRF={crf} encoding timed out")
            return None, 0.0, 0.0
        except Exception as e:
            print(f"[ENCODE] Exception for CRF={crf}: {e}")
            return None, 0.0, 0.0
        finally:
            self.config = saved_config

    def find_optimal_crf(self, input_video: str) -> Tuple[int, Optional[str], float, float]:
      
        """Find CRF with best balance of VMAF and compression using smart scoring"""
        opt_params = self.config.get("optimization_params", {})
        crf_min = opt_params.get("crf_min", 14)
        crf_max = opt_params.get("crf_max", 35)
        vmaf_threshold = opt_params.get("vmaf_threshold", 95)
        compression_threshold = opt_params.get("compression_ratio_target", 10)
    
        print(f"[SEARCH] Looking for VMAF ≥{vmaf_threshold} and compression ≥{compression_threshold}x")
        print(f"[SEARCH] CRF range: {crf_min} to {crf_max}")
        print(f"[SEARCH] Using smart scoring: prioritizes VMAF ≥95, then maximizes compression\n")
    
        best_score = -float('inf')
        best_vmaf = -1.0
        best_compression = 0.0
        best_crf = crf_min
        best_file = None
        
        results = []  # Store all results for analysis
    
        # Search through all CRFs
        for crf in range(crf_min, crf_max + 1):
            output_file, vmaf_score, compression_ratio = self._encode_with_crf(input_video, crf)
            
            if not output_file:
                continue
    
            # ============================================
            # SMART SCORING FORMULA
            # ============================================
            # Goals:
            # 1. Strongly prefer VMAF ≥ 95
            # 2. Among VMAF ≥ 95, maximize compression
            # 3. If VMAF < 95, still consider but with heavy penalty
            
            # Component 1: VMAF Score (exponential reward above 95)
            if vmaf_score >= vmaf_threshold:
                vmaf_component = 100 + (vmaf_score - vmaf_threshold) * 2  # Bonus for exceeding target
            else:
                # Heavy penalty for being below threshold
                deficit = vmaf_threshold - vmaf_score
                vmaf_component = 100 * math.exp(-0.5 * deficit)  # Exponential decay
            
            # Component 2: Compression Score (logarithmic to avoid over-compression)
            if compression_ratio >= compression_threshold:
                # Bonus for meeting target, diminishing returns after
                compression_component = 50 + 10 * math.log10(compression_ratio / compression_threshold + 1)
            else:
                # Linear penalty below target
                compression_component = 50 * (compression_ratio / compression_threshold)
            
            # Component 3: Balance penalty (penalize extreme imbalances)
            vmaf_normalized = vmaf_score / 100.0
            compression_normalized = min(compression_ratio / compression_threshold, 2.0) / 2.0
            balance_penalty = -20 * abs(vmaf_normalized - compression_normalized)
            
            # Final score: weighted combination
            score = (
                0.65 * vmaf_component +      # 65% weight on VMAF
                0.30 * compression_component + # 30% weight on compression  
                0.05 * balance_penalty        # 5% weight on balance
            )
            
            results.append({
                'crf': crf,
                'vmaf': vmaf_score,
                'compression': compression_ratio,
                'score': score,
                'file': output_file
            })
            
            print(f"[SCORE] CRF={crf}: VMAF={vmaf_score:.2f}, Compression={compression_ratio:.2f}x, Score={score:.2f}")
            
            # Track best score
            if score > best_score:
                # Clean up previous best file
                if best_file and best_file != output_file and os.path.exists(best_file):
                    try:
                        os.unlink(best_file)
                    except:
                        pass
                
                best_score = score
                best_vmaf = vmaf_score
                best_compression = compression_ratio
                best_crf = crf
                best_file = output_file
            else:
                # Clean up this file since it's not the best
                try:
                    if output_file != best_file:
                        os.unlink(output_file)
                except:
                    pass
            
            # Early stopping: if we have excellent results and score is declining
            if len(results) >= 5:
                recent_scores = [r['score'] for r in results[-3:]]
                if all(recent_scores[i] > recent_scores[i+1] for i in range(len(recent_scores)-1)):
                    # Scores declining for 3 consecutive CRFs
                    if best_vmaf >= vmaf_threshold and best_compression >= compression_threshold * 0.8:
                        print(f"\n[EARLY STOP] Scores declining and good results achieved. Stopping search.")
                        break
    
        # Print summary table
        print("\n" + "="*80)
        print("RESULTS SUMMARY:")
        print("="*80)
        print(f"{'CRF':<6} {'VMAF':<8} {'Compression':<13} {'Score':<10} {'Status':<20}")
        print("-"*80)
        
        for r in results:
            vmaf_ok = "✓" if r['vmaf'] >= vmaf_threshold else "✗"
            comp_ok = "✓" if r['compression'] >= compression_threshold else "✗"
            status = f"VMAF:{vmaf_ok} COMP:{comp_ok}"
            if r['crf'] == best_crf:
                status += " ← SELECTED"
            print(f"{r['crf']:<6} {r['vmaf']:<8.2f} {r['compression']:<13.2f} {r['score']:<10.2f} {status}")
        
        print("="*80)
        
        # Final verdict
        meets_vmaf = best_vmaf >= vmaf_threshold
        meets_compression = best_compression >= compression_threshold
        
        if meets_vmaf and meets_compression:
            print(f"[SUCCESS] ✓ Both targets met!")
        elif meets_vmaf:
            print(f"[PARTIAL] ✓ VMAF target met, but compression {best_compression:.2f}x < {compression_threshold}x")
        elif meets_compression:
            print(f"[PARTIAL] ✓ Compression target met, but VMAF {best_vmaf:.2f} < {vmaf_threshold}")
        else:
            print(f"[WARNING] Neither target fully met. Best compromise selected.")
        
        print(f"[BEST] CRF={best_crf}, VMAF={best_vmaf:.2f}, Compression={best_compression:.2f}x, Score={best_score:.2f}\n")
        
        return best_crf, best_file, best_vmaf, best_compression


    def compress(self, input_video: str, output_video: str) -> bool:
        """Main compression function"""
        self.start_time = time.time()
        print(f"[START] Starting high-quality compression...")
        print(f"[INPUT] {input_video}")
        print(f"[OUTPUT] {output_video}")

        if not self._validate_input(input_video):
            return False

        # Ensure output is MP4
        if not output_video.lower().endswith('.mp4'):
            output_video = str(Path(output_video).with_suffix('.mp4'))
            print(f"[OUTPUT] Forced MP4 extension: {output_video}")

        Path(output_video).parent.mkdir(parents=True, exist_ok=True)

        # Get input info
        info = self._get_video_info(input_video)
        if info:
            fmt = info.get('format', {})
            try:
                size_mb = int(fmt.get('size', 0)) / (1024 * 1024)
                duration = float(fmt.get('duration', 0))
                print(f"[INFO] Input: {size_mb:.1f} MB, {duration:.1f}s")
            except Exception:
                pass

        # Find optimal CRF
        crf, temp_file, vmaf, compression = self.find_optimal_crf(input_video)
        
        if not temp_file or not os.path.exists(temp_file):
            print("[ERROR] No valid output produced")
            return False

        # Move to final location
        try:
            shutil.move(temp_file, output_video)
            end_time = time.time()
            duration = end_time - self.start_time
            
            final_size = os.path.getsize(output_video) / (1024 * 1024)
            print(f"[SUCCESS] Compression completed in {duration:.1f}s")
            print(f"[SUCCESS] Final: {final_size:.1f} MB, CRF={crf}, VMAF={vmaf:.2f}, Compression={compression:.2f}x")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to move output file: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='H.265/HEVC Video Compressor targeting 95+ VMAF and 10x compression')
    parser.add_argument('--input', required=True, help='Input MP4 video file')
    parser.add_argument('--output', required=True, help='Output MP4 video file')
    parser.add_argument('--config', default='config.json', help='Path to config file')
    args = parser.parse_args()

    compressor = H265Compressor(args.config)
    success = compressor.compress(args.input, args.output)
    
    if not success:
        print("[FAILED] Compression failed")
        sys.exit(1)
    else:
        print("[COMPLETE] Compression successful")


if __name__ == '__main__':
    main()
