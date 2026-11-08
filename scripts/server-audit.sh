#!/usr/bin/env bash
# Read-only inventory. Run on the host with Docker socket access.
set -uo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
section() { printf '\n### %s\n' "$*"; }
section 'Time (Asia/Seoul)'
TZ=Asia/Seoul date --iso-8601=seconds
section 'Account / OS / CPU / RAM'
id
cat /etc/os-release
uname -smr
lscpu
free -h
section 'GPU / driver / active compute processes'
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,driver_version,compute_mode --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
section 'Disk / work directory'
df -hT "$root" /tmp /var/lib/docker
df -i "$root" /var/lib/docker
ls -ld "$root" /var/run/docker.sock
section 'Docker / NVIDIA Toolkit'
docker version
docker info --format '{{json .}}' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({k:d.get(k) for k in ("ServerVersion","Containers","ContainersRunning","Images","Driver","CgroupVersion","Runtimes","DefaultRuntime","DockerRootDir","NCPU","MemTotal")},indent=2))'
nvidia-ctk --version
nvidia-container-cli --version
section 'Existing containers (no environment variables or commands)'
docker ps -a --format 'table {{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}'
section 'Existing images'
docker images --digests --format 'table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.Digest}}'
