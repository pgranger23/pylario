"""Second speed pass: neutrino descendants, fixed-size object columns, freezing."""

from __future__ import annotations

import os

import numpy as np
import pytest

from pylario import ArtFile, EventFile
from pylario import streamers as st
from pylario.event import Event, MCParticles

_DATA = os.environ.get("PYLARIO_TEST_DATA", os.environ.get("PYLAR_TEST_DATA", os.environ.get("PYLAREVD_TEST_DATA", "")))
ATMNU = os.environ.get(
    "PYLARIO_ATMNU", os.environ.get("PYLAR_ATMNU", os.path.join(_DATA, "atmnu_radio_reco.root") if _DATA else ""))
needs_atmnu = pytest.mark.skipif(not os.path.exists(ATMNU), reason="sample file absent")


def _reference_descendants(mc, vtx, tol=2.0):
    """The per-particle walk neutrino_track_ids used before it was vectorised."""
    index = {int(t): i for i, t in enumerate(mc.track_id)}
    start = np.array([p[0] if len(p) else (np.nan, np.nan, np.nan) for p in mc.points], float)
    roots = np.arange(len(mc))
    for i in range(len(mc)):
        j, steps = i, 0
        while steps < 64:
            mother = int(mc.mother[j])
            if mother == 0 or mother not in index:
                break
            j = index[mother]
            steps += 1
        roots[i] = j
    d = np.linalg.norm(start[roots] - vtx, axis=1)
    return np.asarray(mc.track_id)[d < tol]


def _event_with(mc, vtx):
    ev = Event.__new__(Event)
    ev._memo = {}
    ev.true_vertex = lambda: vtx
    ev.mc_particles = lambda *a, **k: mc
    return ev


def test_descendants_match_the_per_particle_walk():
    # 1 (at the vertex) -> 2 -> 3; 4 is a primary elsewhere -> 5; 6's mother is
    # not simulated; 7 <-> 8 form a cycle; 9 is a duplicate track id (the later
    # row wins, as in a dict); 10 descends from that duplicate
    vtx = np.zeros(3)
    tid = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10])
    mom = np.array([0, 1, 2, 0, 4, 99, 8, 7, 1, 4, 9])
    far = np.array([500.0, 0, 0])
    starts = [vtx, vtx + 1, vtx + 2, far, far + 1, vtx, far, far, vtx, far, vtx + 3]
    pts = [np.vstack([s, s + 1]) for s in starts]
    mc = MCParticles(label="t", points=pts, pdg=np.zeros(len(tid), np.int32),
                     track_id=tid.astype(np.int32), mother=mom.astype(np.int32),
                     process=[""] * len(tid))
    got = Event.neutrino_track_ids(_event_with(mc, vtx))
    ref = _reference_descendants(mc, vtx)
    assert sorted(got.tolist()) == sorted(ref.tolist())
    # 9 is in through its first row (child of 1); 10 is not: its mother id 9
    # resolves to the later row, a child of the far primary 4
    assert set(got.tolist()) == {1, 2, 3, 6, 9}


@needs_atmnu
def test_descendants_match_on_a_real_event():
    ev = EventFile(ATMNU)[1]
    got = ev.neutrino_track_ids()
    ref = _reference_descendants(ev.mc_particles(min_points=1), ev.true_vertex())
    assert len(got) > 1000 and np.array_equal(np.sort(got), np.sort(ref))


@needs_atmnu
def test_fixed_object_columns_match_the_generic_reader(monkeypatch):
    product = [p for p in ArtFile(ATMNU).find_product("recob::Track") if "pandoraTrack" in p][0]
    fast = ArtFile(ATMNU).read(product, "recob::Track", 1)
    calls = []
    real = st._fixed_objects
    monkeypatch.setattr(st, "_fixed_objects", lambda *a: calls.append(a[1]) or None)
    slow = ArtFile(ATMNU).read(product, "recob::Track", 1)
    assert any("Cartesian3D" in c for c in calls)        # the bulk path was on offer
    assert len(fast["fTraj"]) == len(slow["fTraj"]) > 0
    for a, b in zip(fast["fTraj"], slow["fTraj"]):
        for key in ("fPositions", "fMomenta"):
            assert len(a[key]) == len(b[key])
            for p, q in zip(a[key], b[key]):
                assert p == q and all(type(v) is float for v in p.values())
    assert real is not None


@needs_atmnu
def test_trajectories_still_frozen():
    mc = EventFile(ATMNU)[1].mc_particles()
    assert all(not p.flags.writeable for p in mc.points[:1000])
    with pytest.raises(ValueError):
        mc.points[0][0, 0] = 1.0
