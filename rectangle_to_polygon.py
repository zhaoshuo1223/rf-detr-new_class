"""
将 LabelMe 风格 JSON 中 shapes[].shape_type 从 rectangle 改为 polygon（不改动 points）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 默认标注目录；也可用命令行: python rectangle_to_polygon.py <json_dir>
json_dir = r"D:\aotto\budingban\data_set\yunqi\0408"


def convert_file(path: Path) -> int:
    """返回本文件内被修改的 shape 数量。"""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    changed = 0

    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        return 0

    for sh in shapes:
        if not isinstance(sh, dict):
            continue
        if sh.get("shape_type") == "rectangle":
            sh["shape_type"] = "polygon"
            changed += 1

    if changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def main() -> None:
    root = Path(json_dir).resolve()
    if len(sys.argv) > 1:
        root = Path(sys.argv[1]).resolve()

    if not root.is_dir():
        print(f"目录不存在: {root}", file=sys.stderr)
        sys.exit(1)

    files = sorted(root.rglob("*.json"))
    total_shapes = 0
    files_touched = 0
    for fp in files:
        n = convert_file(fp)
        if n:
            files_touched += 1
            total_shapes += n
            print(f"{fp}: {n} shape(s) rectangle -> polygon")

    print(f"完成: 处理 JSON 文件 {len(files)} 个，改写 {total_shapes} 个 rectangle，涉及 {files_touched} 个文件。")


if __name__ == "__main__":
    main()
