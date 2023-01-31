# -*- coding: utf-8 -*-
import hashlib
from pathlib import Path


def md5sum(path):
    """
    Calc the MD5 hash of a binary file
    """
    with open(path, mode="rb") as f:
        d = hashlib.md5()
        while True:
            buf = f.read(4096)
            if not buf:
                break
            d.update(buf)
        return d.hexdigest()


def export_list(data: list, output: Path):
    """
    Export a list of elements to a specified location, one element per line.
    """
    with output.open("w") as writer:
        writer.write("\n".join(str(value) for value in data))
