"""
打印文件夹树形结构。用法:
  python test.py                    # 当前目录
  python test.py E:\\path\\to\\dir  # 指定目录
  python test.py . 3                # 限制深度为 3
"""

from __future__ import annotations

import sys
from pathlib import Path


def tree(
    root: Path,
    prefix: str = "",
    max_depth: int | None = None,
    current_depth: int = 0,
) -> None:
    root = root.resolve()
    if not root.is_dir():
        print(f"不是文件夹: {root}", file=sys.stderr)
        sys.exit(1)

    if max_depth is not None and current_depth > max_depth:
        return

    try:
        entries = sorted(
            root.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
    except PermissionError as e:
        print(f"{prefix}[无法访问: {e}]", file=sys.stderr)
        return

    for i, path in enumerate(entries):
        is_last = i == len(entries) - 1
        # ASCII 树形符号，避免 Windows 控制台编码问题
        branch = "`-- " if is_last else "|-- "
        name = path.name + ("/" if path.is_dir() else "")
        print(f"{prefix}{branch}{name}")

        if path.is_dir():
            extension = "    " if is_last else "|   "
            tree(
                path,
                prefix + extension,
                max_depth,
                current_depth + 1,
            )


def main() -> None:
    argv = sys.argv[1:]
    root = Path(argv[0]) if argv else Path.cwd()
    max_depth: int | None = None
    if len(argv) >= 2:
        try:
            max_depth = int(argv[1])
        except ValueError:
            print("第二个参数应为整数（最大深度）", file=sys.stderr)
            sys.exit(1)

    root = root.resolve()
    print(f"{root.name}/")
    tree(root, "", max_depth, 1)


if __name__ == "__main__":
    main()
