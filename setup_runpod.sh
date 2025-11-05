#!/usr/bin/env bash
set -euo pipefail
echo "=== START: setup_ffmpeg_vmaf_svt.sh ==="

# -------- 0) Helpers --------
require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing $1"; exit 1; }; }
log() { echo -e "\033[1;34m[INFO]\033[0m $*"; }
err() { echo -e "\033[1;31m[ERR]\033[0m  $*" >&2; }

# -------- 1) System deps --------
export DEBIAN_FRONTEND=noninteractive
apt update -y
DEPS="git curl ca-certificates build-essential pkg-config yasm nasm meson ninja-build cmake \
      libx264-dev libx265-dev libnuma-dev libvpx-dev libaom-dev \
      libfreetype6-dev libfribidi-dev libass-dev libmp3lame-dev libopus-dev \
      python3 python3-venv python3-pip xxd"
log "Installing: $DEPS"
apt install -y --no-install-recommends $DEPS

require_cmd gcc
require_cmd g++
require_cmd meson
require_cmd ninja
require_cmd cmake

mkdir -p /usr/local/src
cd /usr/local/src

# -------- 2) Build & install libvmaf --------
if [ ! -d vmaf ]; then
  log "Cloning netflix/vmaf ..."
  git clone --depth=1 https://github.com/Netflix/vmaf.git
fi
cd /usr/local/src/vmaf/libvmaf
log "Configuring libvmaf (built-in models, float) ..."
if [ -d build ]; then
  meson setup --reconfigure build --buildtype release -Dbuilt_in_models=true -Denable_float=true
else
  meson setup build --buildtype release -Dbuilt_in_models=true -Denable_float=true
fi
log "Building libvmaf ..."
ninja -C build
log "Installing libvmaf ..."
ninja -C build install
ldconfig

# -------- 3) Build & install SVT-AV1 (libsvtav1) --------
cd /usr/local/src
if [ ! -d SVT-AV1 ]; then
  log "Cloning SVT-AV1 ..."
  git clone --depth=1 https://gitlab.com/AOMediaCodec/SVT-AV1.git SVT-AV1 || \
  git clone --depth=1 https://github.com/AOMediaCodec/SVT-AV1.git SVT-AV1
fi
cd SVT-AV1
rm -rf build && mkdir build && cd build
log "Configuring SVT-AV1 ..."
cmake .. -DCMAKE_BUILD_TYPE=Release
log "Building SVT-AV1 ..."
make -j"$(nproc)"
log "Installing SVT-AV1 ..."
make install
ldconfig

# -------- 4) Build FFmpeg with BOTH libvmaf + libsvtav1 --------
cd /usr/local/src
if [ ! -d FFmpeg ]; then
  log "Cloning FFmpeg ..."
  git clone --depth=1 https://github.com/FFmpeg/FFmpeg.git
fi
cd FFmpeg

# Make sure pkg-config finds libs in /usr/local
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"
# Make sure runtime finds /usr/local libs
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

log "Configuring FFmpeg with libvmaf and libsvtav1 ..."
./configure --prefix=/opt/ffmpeg \
  --extra-cflags="-I/usr/local/include" \
  --extra-ldflags="-L/usr/local/lib -Wl,-rpath,/usr/local/lib -Wl,-rpath,/usr/local/lib/x86_64-linux-gnu" \
  --bindir=/opt/ffmpeg/bin \
  --enable-gpl \
  --enable-libx264 --enable-libx265 \
  --enable-libvpx --enable-libaom \
  --enable-libass --enable-libfreetype \
  --enable-libmp3lame --enable-libopus \
  --enable-libvmaf \
  --enable-libsvtav1

log "Building FFmpeg ..."
make -j"$(nproc)"
log "Installing FFmpeg ..."
make install
ldconfig

# Put /opt/ffmpeg/bin FIRST on PATH for all future shells
echo 'export PATH=/opt/ffmpeg/bin:$PATH' >/etc/profile.d/ffmpeg.sh
chmod +x /etc/profile.d/ffmpeg.sh
# Also update current shell
export PATH=/opt/ffmpeg/bin:$PATH

# -------- 5) Optional: VMAF models in a canonical location --------
log "Making VMAF model files available under /usr/share/vmaf/model ..."
mkdir -p /usr/share/vmaf
if [ ! -d /usr/share/vmaf/model ]; then
  git clone --depth=1 https://github.com/Netflix/vmaf.git /usr/share/vmaf
fi
mkdir -p /usr/share/model
ln -sf /usr/share/vmaf/model/vmaf_v0.6.1.json        /usr/share/model/vmaf_v0.6.1.json
ln -sf /usr/share/vmaf/model/vmaf_float_v0.6.1.json  /usr/share/model/vmaf_float_v0.6.1.json
ln -sf /usr/share/vmaf/model/vmaf_b_v0.6.3.json      /usr/share/model/vmaf_b_v0.6.3.json
ln -sf /usr/share/vmaf/model/vmaf_float_b_v0.6.3.json /usr/share/model/vmaf_float_b_v0.6.3.json

# -------- 6) Verify the RIGHT binary and features --------
log "Verifying (must show /opt/ffmpeg/bin/ffmpeg first):"
type -a ffmpeg || true
/opt/ffmpeg/bin/ffmpeg -hide_banner -version | grep -- --enable-libvmaf || { err "libvmaf missing in configure"; exit 1; }
/opt/ffmpeg/bin/ffmpeg -hide_banner -version | grep -- --enable-libsvtav1 || { err "libsvtav1 missing in configure"; exit 1; }
/opt/ffmpeg/bin/ffmpeg -hide_banner -filters  | grep -i libvmaf || { err "libvmaf filter missing"; exit 1; }
/opt/ffmpeg/bin/ffmpeg -hide_banner -encoders | grep -i svt     || { err "SVT-AV1 encoder missing"; exit 1; }

log "All good ✅  (FFmpeg has BOTH libvmaf and libsvtav1)"

echo "=== DONE: setup_ffmpeg_vmaf_svt.sh ==="
