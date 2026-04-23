"""
Merge JPEGImages/json from second-level leaves under root_dir into target_dir.

Rule:
- Keep original file names.
- If a filename is duplicated in sources, that name is excluded entirely.
- target_dir must not contain any excluded duplicate name.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

root_dir = r"D:\aotto\dingweixiao\aaa_use_data\aaa_use_data"
image_dir = "JPEGImages"
json_dir = "json"

target_dir = r"D:\aotto\dingweixiao\aaa_origin_data"


def main() -> None:
    root = Path(root_dir)
    out_root = Path(target_dir)
    out_images = out_root / image_dir
    out_jsons = out_root / json_dir
    out_images.mkdir(parents=True, exist_ok=True)
    out_jsons.mkdir(parents=True, exist_ok=True)

    n_sites = 0
    n_img = 0
    n_json = 0
    n_removed_dup_img = 0
    n_removed_dup_json = 0

    if not root.is_dir():
        raise FileNotFoundError(f"root_dir 不存在: {root}")

    img_sources: dict[str, list[Path]] = defaultdict(list)
    json_sources: dict[str, list[Path]] = defaultdict(list)

    for level1 in sorted(root.iterdir()):
        if not level1.is_dir():
            continue
        for level2 in sorted(level1.iterdir()):
            if not level2.is_dir():
                continue
            src_imgs = level1 / level2 / image_dir
            src_j = level1 / level2 / json_dir
            if not src_imgs.is_dir() or not src_j.is_dir():
                continue
            n_sites += 1

            for f in src_imgs.iterdir():
                if not f.is_file():
                    continue
                img_sources[f.name].append(f)

            for f in src_j.iterdir():
                if not f.is_file():
                    continue
                json_sources[f.name].append(f)

    # Handle images: copy unique names only; remove duplicate names from target if present.
    for name, src_list in img_sources.items():
        dest_img = out_images / name
        if len(src_list) > 1:
            if dest_img.exists():
                dest_img.unlink()
                n_removed_dup_img += 1
            print(f"[drop image] 重名文件不保留: {name!r}, 出现{len(src_list)}次")
            continue
        shutil.copy2(src_list[0], dest_img)
        n_img += 1

    # Handle json: copy unique names only; remove duplicate names from target if present.
    for name, src_list in json_sources.items():
        dest_j = out_jsons / name
        if len(src_list) > 1:
            if dest_j.exists():
                dest_j.unlink()
                n_removed_dup_json += 1
            print(f"[drop json] 重名文件不保留: {name!r}, 出现{len(src_list)}次")
            continue
        shutil.copy2(src_list[0], dest_j)
        n_json += 1

    print(
        f"完成: 含 {image_dir!r}+{json_dir!r} 的目录数 = {n_sites}, "
        f"复制图片 = {n_img}, 复制 json = {n_json}, "
        f"移除目标中重名 图片 = {n_removed_dup_img}, json = {n_removed_dup_json} -> {out_root}"
    )


if __name__ == "__main__":
    main()
