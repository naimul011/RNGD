"""Quick CLI: python rvt_cli.py [info|preview|streams|search|categories] [file] [term]"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import rvt_tools as rt

HERE = Path(__file__).parent


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"
    files = [Path(sys.argv[2])] if len(sys.argv) > 2 and sys.argv[2].endswith(".rvt") else rt.find_rvts(HERE / "extracted")
    for f in files:
        print(f"\n=== {f.name}  ({f.parent.name})")
        if cmd == "info":
            i = rt.basic_info(f)
            print({k: i[k] for k in ("revit_version", "build", "original_path", "file_size_mb")})
        elif cmd == "preview":
            out = rt.extract_preview(f, HERE / "previews" / f"{f.parent.name}.png")
            print("preview ->", out)
        elif cmd == "streams":
            for s in rt.list_streams(f):
                print(f"  {s['stream']:32s} {s['bytes']:>12,}")
        elif cmd == "categories":
            for k, v in list(rt.category_counts(f).items())[:12]:
                print(f"  {k:14s} {v}")
        elif cmd == "search":
            term = sys.argv[-1]
            for s, n in rt.search_strings(f, term):
                print(f"  {n:5d}  {s}")


if __name__ == "__main__":
    main()
