"""Tests for pylario: art-ROOT reader, data model, and geometry."""

from __future__ import annotations

import os
import numpy as np
import pytest

from pylario import (
    ArtFile, ArtReadError, EventFile, Geometry, GeometryError,
    Hits, SpacePoints, Tracks, Showers, Vertices, OpticalActivity,
    TruthDeposits, MCParticles, Neutrino, PrimaryInteraction, physics
)

_DATA = os.environ.get("PYLARIO_TEST_DATA", os.environ.get("PYLAR_TEST_DATA", os.environ.get("PYLAREVD_TEST_DATA", "")))
ROCKMU = os.environ.get(
    "PYLARIO_ROCKMU", os.environ.get("PYLAR_ROCKMU", os.path.join(_DATA, "rock_muons_reco1.root") if _DATA else ""))
ATMNU = os.environ.get(
    "PYLARIO_ATMNU", os.environ.get("PYLAR_ATMNU", os.path.join(_DATA, "atmnu_radio_reco.root") if _DATA else ""))
GEOM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "pylario", "geom", "dune10kt_v6_1x2x6.npz")

needs_data = pytest.mark.skipif(not os.path.exists(ROCKMU), reason="sample file absent")
needs_geom = pytest.mark.skipif(not os.path.exists(GEOM), reason="geometry absent")


# ---- layout & low-level artio ----------------------------------------------

@needs_data
@pytest.mark.parametrize("classname,itemsize", [
    ("recob::Hit", 109),
    ("sim::SimEnergyDeposit", 132),
    ("recob::SpacePoint", 44),
])
def test_record_itemsize(classname, itemsize):
    assert ArtFile(ROCKMU).layout(classname).itemsize == itemsize


@needs_data
def test_wireid_nested_headers():
    lay = ArtFile(ROCKMU).layout("recob::Hit")
    blk = next(b for b in lay.blocks if b.name == "fWireID")
    assert blk.itemsize == 41
    assert not blk.is_flat
    assert [c.name for c in blk.columns] == [
        "fWireID.isValid", "fWireID.Cryostat", "fWireID.TPC",
        "fWireID.Plane", "fWireID.Wire"]


@needs_data
def test_hits_decode_known_values():
    h = ArtFile(ROCKMU).read("recob::Hits_hitfd__Reco1.", "recob::Hit", 0)
    assert len(h["fChannel"]) == 6671
    assert h["fChannel"][0] == 20548
    assert h["fPeakTime"][0] == pytest.approx(2627.44, abs=0.01)
    assert h["fWireID.Wire"][0] == 868
    assert h["fWireID.Plane"][0] == 0
    assert h["fWireID.TPC"][0] == 16


@needs_data
def test_event_ids():
    ids = ArtFile(ROCKMU).event_ids()
    assert len(ids) == 10
    assert tuple(ids[0]) == (1, 0, 1)
    assert tuple(ids[9]) == (1, 0, 10)


def test_unreadable_file_raises_artreaderror(tmp_path):
    bad = tmp_path / "corrupt.root"
    bad.write_bytes(b"not a root file")
    with pytest.raises(ArtReadError):
        ArtFile(str(bad))


# ---- geometry --------------------------------------------------------------

@needs_geom
def test_geometry_covers_all_channels():
    g = Geometry(GEOM)
    assert g.detector == "dune10kt_v6_1x2x6"
    assert g.nchannels == 30720
    assert len(np.unique(g.w_chan)) == 30720
    assert g.has_drift


def test_malformed_geometry_raises_geometryerror(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez(bad, detector="dummy")
    with pytest.raises(GeometryError, match="missing"):
        Geometry(str(bad))


@needs_data
@needs_geom
def test_channel_map_matches_reconstruction():
    f = EventFile(ROCKMU, geometry=GEOM)
    h = f[0].hits()
    rows = f.geometry.wire_rows(h.cryo, h.tpc, h.plane, h.wire)
    assert (rows >= 0).all()
    assert np.array_equal(f.geometry.w_chan[rows], h.channel)


# ---- data models & slicing -------------------------------------------------

def test_hit_selection_slicing():
    n = 10
    h = Hits(
        product="recob::Hits_test",
        channel=np.arange(n, dtype=np.int32),
        tick=np.linspace(100, 200, n),
        start_tick=np.zeros(n, dtype=np.int32),
        end_tick=np.ones(n, dtype=np.int32) * 10,
        integral=np.ones(n) * 50.0,
        amplitude=np.ones(n) * 15.0,
        rms=np.ones(n),
        multiplicity=np.ones(n, dtype=np.int32),
        goodness=np.ones(n),
        cryo=np.zeros(n, dtype=np.int32),
        tpc=np.zeros(n, dtype=np.int32),
        plane=np.zeros(n, dtype=np.int32),
        wire=np.arange(n, dtype=np.int32),
        valid=np.ones(n, dtype=bool),
        w=np.linspace(0, 100, n),
        x=np.linspace(0, 200, n),
        view=np.zeros(n, dtype=np.int32),
        drift_sign=np.ones(n, dtype=np.int32),
        panel=np.zeros(n, dtype=np.int32),
        orientation=np.zeros(n, dtype=np.int64),
        chan_view=np.zeros(n, dtype=np.int32),
    )
    assert len(h) == 10
    sub = h[h.channel < 5]
    assert len(sub) == 5
    assert list(sub.channel) == [0, 1, 2, 3, 4]
    assert sub.label == "recob::Hits_test [5 hits]"


def test_tracks_ragged_selection():
    pts = [np.zeros((5, 3)), np.ones((10, 3))]
    trks = Tracks(
        label="recob::Tracks_test",
        points=pts,
        id=np.array([1, 2], dtype=np.int32),
        chi2=np.array([0.5, 0.8]),
        ndof=np.array([10, 20]),
    )
    assert len(trks) == 2
    sub = trks[[1]]
    assert len(sub) == 1
    assert len(sub.points[0]) == 10


def test_display_without_pylarevd_raises_importerror():
    """Event.display() should give a clean actionable error when pylarevd is missing."""
    import sys
    saved = sys.modules.get("pylarevd")
    sys.modules["pylarevd"] = None
    try:
        from pylario.event import Event
        ev = Event(None, 0)
        with pytest.raises(ImportError, match="requires the 'pylarevd' visualization package"):
            ev.display()
    finally:
        if saved is not None:
            sys.modules["pylarevd"] = saved
        else:
            sys.modules.pop("pylarevd", None)


def test_read_memberwise_column_empty():
    """An empty vector (n=0) has no column prefix on disk and must yield an empty list."""
    from pylario import streamers as st
    cur = st.Cursor(b"", 0)
    assert st.read_memberwise_column(None, None, cur, 0) == []
