"""TEDE command-line interface (Ultralytics-style key=value arguments).

Usage:
    tede train data=data.yaml arch=retinanet epochs=50 batch=8
    tede val   model=runs/.../best.pt data=data.yaml
    tede predict model=runs/.../best.pt source=img.jpg
    tede export  model=runs/.../best.pt format=onnx
    tede serve   model=runs/.../best.pt port=8000
    tede preprocess source=raw_data/ output=data/ nc=3
    tede registry list
    tede --version
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from tede import __version__


def _coerce(v: str) -> Any:
    """Coerce a string CLI value to bool / None / int / float / str."""
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("none", "null"):
        return None
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return v


def parse_kv(args: List[str]) -> Dict[str, Any]:
    """Parse a list of ``key=value`` tokens into a dict."""
    out: Dict[str, Any] = {}
    for arg in args:
        if "=" not in arg:
            raise SystemExit(f"Invalid argument '{arg}': expected key=value")
        k, v = arg.split("=", 1)
        out[k.strip()] = _coerce(v.strip())
    return out


# ---------- subcommands ----------

def cmd_train(kv: Dict[str, Any]) -> None:
    from tede import TEDE

    if "data" not in kv:
        raise SystemExit("train requires data=path/to/data.yaml")
    arch = str(kv.pop("arch", "retinanet"))
    weights = kv.pop("model", None)  # optional resume checkpoint
    model = TEDE(weights=weights, arch=arch)
    if weights:
        kv.setdefault("resume", True)
    best = model.train(**kv)
    print(f"\nBest checkpoint: {best}")


def cmd_val(kv: Dict[str, Any]) -> None:
    from tede import TEDE

    weights = kv.pop("model", None)
    if not weights:
        raise SystemExit("val requires model=best.pt")
    if "data" not in kv:
        raise SystemExit("val requires data=path/to/data.yaml")
    metrics = TEDE(weights=weights).val(**kv)
    summary = {k: v for k, v in metrics.items() if k != "per_class"}
    print(json.dumps(summary, indent=2))


def cmd_predict(kv: Dict[str, Any]) -> None:
    from tede import TEDE

    weights = kv.pop("model", None)
    if not weights:
        raise SystemExit("predict requires model=best.pt")
    if "source" not in kv:
        raise SystemExit("predict requires source=path/to/img_or_dir")
    source = kv.pop("source")
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    results = TEDE(weights=weights).predict(source=source, **kv)
    print(json.dumps(results, indent=2))


def cmd_export(kv: Dict[str, Any]) -> None:
    from tede import TEDE

    weights = kv.pop("model", None)
    if not weights:
        raise SystemExit("export requires model=best.pt")
    fmt = str(kv.pop("format", "onnx"))
    artifact = TEDE(weights=weights).export(format=fmt, **kv)
    print(f"Exported: {artifact}")


def cmd_serve(kv: Dict[str, Any]) -> None:
    from tede import TEDE

    weights = kv.pop("model", None)
    if not weights:
        raise SystemExit("serve requires model=best.pt")
    host = str(kv.pop("host", "0.0.0.0"))
    port = int(kv.pop("port", 8000))
    TEDE(weights=weights).serve(host=host, port=port)


def cmd_preprocess(kv: Dict[str, Any]) -> None:
    from tede.data import run as run_preprocess

    if "source" not in kv:
        raise SystemExit("preprocess requires source=raw_data/")
    run_preprocess(
        source=Path(str(kv["source"])),
        output=Path(str(kv.get("output", "data"))),
        val_frac=float(kv.get("val", 0.2)),
        test_frac=float(kv.get("test", 0.1)),
        num_classes=int(kv["nc"]) if "nc" in kv else None,
        seed=int(kv.get("seed", 42)),
    )


def cmd_registry(kv: Dict[str, Any]) -> None:
    from tede.registry import ModelRegistry

    action = kv.pop("action", "list") if kv else "list"
    reg = ModelRegistry()
    if action == "list":
        print(json.dumps(reg.list_versions(), indent=2))
    elif action == "best":
        print(json.dumps(reg.get_best(), indent=2))
    elif action == "compare":
        version = kv.get("version")
        if not version:
            raise SystemExit("registry compare requires version=v1.0")
        print(json.dumps(reg.compare_with_best(str(version)), indent=2))
    else:
        raise SystemExit(f"Unknown registry action: {action}")


COMMANDS = {
    "train": cmd_train,
    "val": cmd_val,
    "predict": cmd_predict,
    "export": cmd_export,
    "serve": cmd_serve,
    "preprocess": cmd_preprocess,
    "registry": cmd_registry,
}


def main() -> None:
    """CLI entrypoint. Installed as the ``tede`` console script."""
    parser = argparse.ArgumentParser(
        prog="tede",
        description="TEDE — independent production object detection framework.",
        usage="tede <command> key=value [key=value ...]",
    )
    parser.add_argument("--version", action="version", version=f"tede {__version__}")
    parser.add_argument("command", nargs="?", choices=list(COMMANDS.keys()), help="Subcommand to run")
    parser.add_argument("kv", nargs=argparse.REMAINDER, help="key=value arguments")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    kv: Dict[str, Any] = {}
    extra = list(args.kv)
    if args.command == "registry" and extra and "=" not in extra[0]:
        kv["action"] = extra.pop(0)
    kv.update(parse_kv(extra))

    try:
        COMMANDS[args.command](kv)
    except KeyboardInterrupt:
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[tede {args.command}] failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
