#!/usr/bin/env bash
# === Activate Python + FFmpeg environment ===

# 1️⃣  Load FFmpeg environment (adds /opt/ffmpeg/bin to PATH)
if [ -f /etc/profile.d/ffmpeg.sh ]; then
    source /etc/profile.d/ffmpeg.sh
fi

# 2️⃣  Activate Python virtual environment
source /workspace/venv/bin/activate

# 3️⃣  Set runtime variables
export PYTHONUNBUFFERED=1
export LIBVMAF_MODEL_PATH="/usr/share/vmaf/model"
export VMAF_MODEL="/usr/share/vmaf/model/vmaf_v0.6.1.json"

# 4️⃣  Confirm the active FFmpeg binary
echo "Active ffmpeg path:"
type -a ffmpeg | head -n 3
ffmpeg -hide_banner -version | grep -- --enable-libvmaf
ffmpeg -hide_banner -version | grep -- --enable-libsvtav1
echo "✅ Environment ready!"
