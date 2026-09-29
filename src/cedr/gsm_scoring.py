#!/usr/bin/env python
"""Score GSM8K generations by extracting and comparing final numeric answers."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?")
BOXED_RE = re.compile(
    r"\\boxed\s*\{\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?)"
)


def normalize_number(value):
    if value is None:
        return None
    text = str(value).strip().replace("$", "").replace(",", "")
    text = text.rstrip(".。,) ]}")
    if not text:
        return None
    try:
        if "/" in text:
            frac = Fraction(text)
            return Decimal(frac.numerator) / Decimal(frac.denominator)
        return Decimal(text)
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return text


def extract_gold(text):
    if text is None:
        return None
    if "####" in text:
        tail = text.split("####")[-1]
        match = NUM_RE.search(tail)
        return normalize_number(match.group(0)) if match else normalize_number(tail)
    nums = NUM_RE.findall(text)
    return normalize_number(nums[-1]) if nums else None


def extract_pred(text):
    if not text:
        return None
    boxed = BOXED_RE.findall(text)
    if boxed:
        return normalize_number(boxed[-1])
    markers = ["The answer is:", "The answer is", "####", "Answer:", "answer is"]
    for marker in markers:
        if marker in text:
            tail = text.split(marker)[-1]
            match = NUM_RE.search(tail)
            if match:
                return normalize_number(match.group(0))
    nums = NUM_RE.findall(text)
    return normalize_number(nums[-1]) if nums else None
