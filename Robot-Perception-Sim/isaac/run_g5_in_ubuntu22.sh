#!/usr/bin/env bash
# Isolated user-owned Ubuntu 22 userspace for this PoC only; no host OS/driver install.
set -euo pipefail
poc=/home/lizhengran/isaacsim-workspace/g5_static
deps=/home/lizhengran/isaacsim-workspace/g2_zed_poc
g3=/home/lizhengran/isaacsim-workspace/g3_acquisition
rootfs="$poc/ubuntu22-rootfs"
isaac=/home/lizhengran/software/isaac-sim-4.5.0
[[ -f "$rootfs/etc/os-release" && -d "$isaac" && $# -gt 0 ]] || {
  echo 'Requires the prepared PoC rootfs, existing Isaac 4.5, and a command.' >&2; exit 2;
}
[[ "$(readlink -f "$rootfs")" == "$rootfs" && "$(readlink -f "$poc")" == "$poc" ]] || {
  echo 'Unexpected PoC/rootfs symlink; refusing mounts.' >&2; exit 2;
}
mkdir -p "$poc/kit-cache"
mounts=(--unshare-user --uid 0 --gid 0 --unshare-pid --die-with-parent
  --bind "$rootfs" / --ro-bind "$g3" /mnt
  --ro-bind "$isaac" /isaac --ro-bind "$deps" /opt --bind "$poc" /poc
  --bind "$poc/kit-cache" /isaac/kit/cache
  --dev-bind /dev /dev --proc /proc --ro-bind /proc/driver/nvidia /proc/driver/nvidia
  --ro-bind /sys /sys
  --tmpfs /tmp --ro-bind /etc/resolv.conf /etc/resolv.conf
  --ro-bind /etc/hosts /etc/hosts --chdir /poc)
# Map only host NVIDIA driver libraries, never host libc/libstdc++ or its whole lib directory.
shopt -s nullglob
for lib in /usr/lib/x86_64-linux-gnu/libcuda.so* \
           /usr/lib/x86_64-linux-gnu/libnvoptix.so* \
           /usr/lib/x86_64-linux-gnu/libnvcuvid.so* \
           /usr/lib/x86_64-linux-gnu/libnvidia-*.so* \
           /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so* \
           /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so* \
           /usr/lib/x86_64-linux-gnu/libGLESv1_CM_nvidia.so* \
           /usr/lib/x86_64-linux-gnu/libGLESv2_nvidia.so*; do
  [[ -f "$lib" ]] && mounts+=(--ro-bind "$(readlink -f "$lib")" "$lib")
done
for config in /usr/share/vulkan/icd.d/nvidia_icd.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json; do
  [[ -f "$config" ]] && mounts+=(--ro-bind "$config" "$config")
done
# Preserve HOME/CODEX_HOME. /home/lizhengran exists only inside the private rootfs.
exec /usr/bin/bwrap "${mounts[@]}" "$@"
