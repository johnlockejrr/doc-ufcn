#!/usr/bin/env python
# -*- coding: utf-8 -*-
import hashlib


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
