import contextlib
import csv
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import shutil
import tempfile


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def integer(value, label, minimum=0):
    if isinstance(value, bool):
        raise ValueError(label + " must be an integer")
    try:
        exact = Decimal(str(value))
        if not exact.is_finite() or exact != exact.to_integral_value() or exact < minimum:
            raise ValueError()
        result = int(exact)
    except (ValueError, TypeError, OverflowError, InvalidOperation):
        raise ValueError(label + " must be an integer >= " + str(minimum))
    return result


def finite(value, label):
    if isinstance(value, bool):
        raise ValueError(label + " must be a finite number")
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(label + " must be a finite number")
    if not math.isfinite(result):
        raise ValueError(label + " must be finite")
    return result


def separated(input_root, output):
    source, destination = Path(input_root).resolve(), Path(output).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Input and output must be separate and must not contain each other")


def within(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Expected a safe relative path: " + str(relative))
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("Path escapes its root: " + str(relative))
    return target


@contextlib.contextmanager
def new_output(output, input_root=None):
    """Commit only a complete new directory; never overwrite existing output."""
    destination = Path(output).expanduser().absolute()
    if input_root is not None:
        separated(input_root, destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Output already exists; choose a new directory: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + destination.name + "-", dir=str(destination.parent)))
    try:
        yield temporary
        if destination.exists() or destination.is_symlink():
            raise ValueError("Output appeared during conversion; refusing overwrite")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
