"""The vectorised member-wise column readers agree with the generic ones."""

from __future__ import annotations

import os

import numpy as np
import pytest

from pylario import ArtFile
from pylario import streamers as st

_DATA = os.environ.get("PYLARIO_TEST_DATA", os.environ.get("PYLAR_TEST_DATA", os.environ.get("PYLAREVD_TEST_DATA", "")))
ATMNU = os.environ.get(
    "PYLARIO_ATMNU", os.environ.get("PYLAR_ATMNU", os.path.join(_DATA, "atmnu_radio_reco.root") if _DATA else ""))
needs_atmnu = pytest.mark.skipif(not os.path.exists(ATMNU), reason="sample file absent")
PRODUCT = "simb::MCParticles_largeant__G4."


def _decode(monkeypatch=None, fast=True):
    if not fast:
        monkeypatch.setattr(st, "_is_trajectory_class", lambda *a: False)
        monkeypatch.setattr(st, "_primitive_runs", lambda *a: None)
    return ArtFile(ATMNU).read(PRODUCT, "simb::MCParticle", 1)


def _steps(t):
    s = t["ftrajectory"]
    if isinstance(s, np.ndarray):
        return s
    if not len(s):
        return np.zeros((0, 8))
    return np.array([[a["fP.fX"], a["fP.fY"], a["fP.fZ"], a["fE"],
                      b["fP.fX"], b["fP.fY"], b["fP.fZ"], b["fE"]] for a, b in s])


@needs_atmnu
def test_trajectories_daughters_strings_match_generic(monkeypatch):
    fast = _decode()
    assert isinstance(fast["ftrajectory"][0]["ftrajectory"], np.ndarray)
    with monkeypatch.context() as m:
        slow = _decode(m, fast=False)
    assert not isinstance(slow["ftrajectory"][0]["ftrajectory"], np.ndarray)
    n = len(slow["fpdgCode"])
    assert n == len(fast["fpdgCode"]) > 30000
    for a, b in zip(fast["ftrajectory"], slow["ftrajectory"]):
        s = _steps(b)
        assert a["ftrajectory"].shape == s.shape and np.array_equal(a["ftrajectory"], s)
        assert [(int(i), int(c)) for i, c in a["fTrajectoryProcess"]] == \
               [(int(i), int(c)) for i, c in b["fTrajectoryProcess"]]
    assert sum(len(t["fTrajectoryProcess"]) for t in fast["ftrajectory"]) > 1000
    for a, b in zip(fast["fdaughters"], slow["fdaughters"]):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    assert fast["fprocess"] == slow["fprocess"] and fast["fendprocess"] == slow["fendprocess"]
    for key in ("fpdgCode", "ftrackId", "fmother", "fGvtx.fE"):
        assert np.array_equal(fast[key], slow[key])


@needs_atmnu
def test_trajectory_column_falls_back_on_unexpected_bytes(monkeypatch):
    seen = {}
    real = st._trajectory_column

    def spy(file, classname, cur, n):
        seen.update(file=file, classname=classname, buf=cur.buf, start=cur.pos, n=n)
        return real(file, classname, cur, n)

    monkeypatch.setattr(st, "_trajectory_column", spy)
    ArtFile(ATMNU).read(PRODUCT, "simb::MCParticle", 1)
    assert seen["classname"] == "simb::MCTrajectory"
    file, start, n = seen["file"], seen["start"], seen["n"]
    assert real(file, "simb::MCTrajectory", st.Cursor(seen["buf"], start), n) is not None
    # flip a bit of the first TLorentzVector's byte count: the fast reader must
    # refuse, leaving the cursor where it was for the generic reader
    buf = bytearray(seen["buf"])
    buf[start + 22 + 3] ^= 0x01
    cur = st.Cursor(bytes(buf), start)
    assert real(file, "simb::MCTrajectory", cur, n) is None
    assert cur.pos == start


def test_pieces_and_primitive_runs():
    arr = np.arange(10)
    assert [p.tolist() for p in st._pieces(arr, np.array([3, 0, 7]))] == \
           [[0, 1, 2], [], [3, 4, 5, 6, 7, 8, 9]]
    # three runs of big-endian ints: [1, 2], [], [7]
    buf = (b"\x00\x00\x00\x02" + b"\x00\x00\x00\x01\x00\x00\x00\x02" + b"\x00\x00\x00\x00"
           + b"\x00\x00\x00\x01" + b"\x00\x00\x00\x07")
    cur = st.Cursor(buf, 0)
    runs = st._primitive_runs(cur, "int", 3, len(buf))
    assert [r.tolist() for r in runs] == [[1, 2], [], [7]] and cur.pos == len(buf)
    cur = st.Cursor(buf, 0)
    assert st._primitive_runs(cur, "int", 3, len(buf) - 1) is None and cur.pos == 0
