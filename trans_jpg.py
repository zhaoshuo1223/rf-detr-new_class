"""
将目录下 bmp 转为 jpg，并更新 JSON 标注里对图片文件名的引用（.bmp -> .jpg）。
"""
from __future__ import annotations

import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("需要 Pillow：pip install Pillow") from e

# 要处理的根目录（可改为命令行参数）
root_dir = r"D:\aotto\budingban\data_set\aaa_hebing"


def _unlink_retry(path: Path, attempts: int = 8, delay_s: float = 0.08) -> None:
    """Windows 上句柄释放或杀毒扫描可能滞后，删除失败时短暂重试。"""
    last: OSError | None = None
    for i in range(attempts):
        try:
            path.unlink()
            return
        except OSError as e:
            last = e
            if i < attempts - 1:
                time.sleep(delay_s)
    if last is not None:
        raise last


def bmp_to_jpg(bmp_path: Path, quality: int = 95) -> Path | None:
    """将单张 bmp 存为同目录下同主文件名的 .jpg，成功后删除 bmp。失败返回 None。"""
    jpg_path = bmp_path.with_suffix(".jpg")
    try:
        # 先整文件读入内存再解码，避免 Pillow 长时间占用 bmp 导致 Windows 上 unlink 报 WinError 32。
        raw = bmp_path.read_bytes()
        with Image.open(BytesIO(raw)) as im:
            im.load()
            rgb = im.convert("RGB")
        rgb.save(jpg_path, format="JPEG", quality=quality, optimize=True)
        _unlink_retry(bmp_path)
        return jpg_path
    except OSError as err:
        print(f"[skip] {bmp_path}: {err}", file=sys.stderr)
        return None


def update_json_file_names(json_path: Path) -> bool:
    """
    递归遍历 JSON，将所有键 file_name 且以 .bmp/.BMP 结尾的值改为 .jpg。
    有修改则写回文件，返回是否修改。
    """
    try:
        text = json_path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        print(f"[skip json] {json_path}: {e}", file=sys.stderr)
        return False

    changed = False

    def visit(obj: object) -> None:
        nonlocal changed
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "imagePath" and isinstance(v, str):
                    base, ext = os.path.splitext(v)
                    if ext.lower() == ".bmp":
                        obj[k] = base + ".jpg"
                        changed = True
                else:
                    visit(v)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(data)
    if changed:
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def main() -> None:
    root = Path(root_dir).resolve()
    if not root.is_dir():
        print(f"目录不存在: {root}", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) > 1:
        root = Path(sys.argv[1]).resolve()

    bmp_files = sorted(root.rglob("*.bmp")) + sorted(root.rglob("*.BMP"))
    # 去重（Windows 上可能大小写重复）
    seen: set[Path] = set()
    unique_bmps: list[Path] = []
    for p in bmp_files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique_bmps.append(p)

    converted = 0
    for bmp in unique_bmps:
        if bmp_to_jpg(bmp) is not None:
            converted += 1

    json_changed = 0
    for jp in sorted(root.rglob("*.json")):
        if update_json_file_names(jp):
            json_changed += 1

    print(f"根目录: {root}")
    print(f"BMP -> JPG 成功: {converted} / {len(unique_bmps)}")
    print(f"已更新 JSON 文件数: {json_changed}")


if __name__ == "__main__":
    main()
