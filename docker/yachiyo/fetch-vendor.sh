#!/usr/bin/env bash
set -euo pipefail
vendor_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/vendor"
mkdir -p "$vendor_root"
fetch() {
    local name="$1" url="$2" commit="$3" target="$vendor_root/$1"
    if [[ -e "$target" ]]; then
        [[ -d "$target/.git" ]] || { echo "Refusing existing non-git path: $target" >&2; exit 1; }
        [[ "$(git -C "$target" rev-parse HEAD)" == "$commit" ]] || {
            echo "Commit mismatch; preserve existing checkout and resolve manually: $target" >&2; exit 1;
        }
        [[ -z "$(git -C "$target" status --porcelain)" ]] || {
            echo "Uncommitted changes; refusing to replace: $target" >&2; exit 1;
        }
        echo "Verified $name $commit"
        return
    fi
    git init "$target"
    git -C "$target" remote add origin "$url"
    git -C "$target" fetch --depth 1 origin "$commit"
    git -C "$target" checkout --detach FETCH_HEAD
    [[ "$(git -C "$target" rev-parse HEAD)" == "$commit" ]]
}
fetch BoT-SORT https://github.com/NirAharon/BoT-SORT.git 251985436d6712aaf682aaaf5f71edb4987224bd
fetch deep-person-reid https://github.com/KaiyangZhou/deep-person-reid.git f8cd150fdf77e8d9e1ed143b7f308c2c609ded50
fetch mmpose https://github.com/open-mmlab/mmpose.git 0d9a00c1d64b68999e4b2a4ee5ed0d5ac2c3e296
