"""ProtoDUNE-HD geometry and the online trigger chain (TPs -> TAs -> TCs)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from pylario import EventFile, Geometry
from pylario.event import TC_TYPES

# e.g. prod_protodunehd_beamneutrino_1_342725_42_1777946041_gen_g4_detsim_trigger_reco_stage1_reco_stage2.root
PDHD = os.environ.get("PYLARIO_PDHD", "")
GEOM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "pylario", "geom", "protodunehdv6.npz")

needs_pdhd = pytest.mark.skipif(not os.path.exists(PDHD), reason="set PYLARIO_PDHD")


@pytest.fixture(scope="module")
def pdhd():
    return EventFile(PDHD)


def test_pdhd_geometry():
    g = Geometry(GEOM)
    assert g.detector == "protodunehdv6"
    assert g.nchannels == 10240
    assert len(g.tpcs) == 8                       # 4 APAs x 2 faces
    assert g.wrapped_channels == 3200 + 3200      # every induction channel


@needs_pdhd
def test_file_picks_its_geometry_and_clock(pdhd):
    assert pdhd.geometry.detector == "protodunehdv6"
    assert pdhd.art.service_config("DetectorClocksService")["ClockSpeedTPC"] == "2"
    # the disambiguated collection, not gaushit
    assert pdhd.resolve_hit_product(None).split("_")[1] == "hitpdune"


@needs_pdhd
def test_trigger_chain_links(pdhd):
    tr = pdhd[3].trigger()
    assert (tr.n_tp, tr.n_ta, tr.n_tc) == (31360, 2, 2)
    assert tr.ticks_per_dts == pytest.approx(2 / 62.5)
    assert [len(t) for t in tr.ta_tps] == [627, 1094]
    assert [list(a) for a in tr.tc_tas] == [[0], [1]]
    assert list(tr.ta_tc) == [0, 1]
    assert [TC_TYPES[int(t)] for t in tr.tc_type] == ["ADCSimpleWindow"] * 2
    for i, tps in enumerate(tr.ta_tps):
        assert np.all(tr.tp_ta[tps] == i)
        # an ADCSimpleWindow activity is the ADC sum of its primitives ...
        assert tr.tp_adc_integral[tps].sum() == pytest.approx(tr.ta_adc_integral[i])
        # ... inside its time window
        assert tr.tp_tick_start[tps].min() >= tr.ta_tick_start[i] - 1e-6
        assert tr.tp_tick_start[tps].max() <= tr.ta_tick_end[i] + 1e-6
    assert (tr.tp_ta >= 0).sum() == 627 + 1094


@needs_pdhd
def test_trigger_without_candidates(pdhd):
    tr = pdhd[0].trigger()
    assert tr.n_tp == 55206 and tr.n_ta == 0 and tr.n_tc == 0
    assert np.all(tr.tp_ta == -1) and tr.ta_tps == [] and tr.tc_tas == []


@needs_pdhd
def test_trigger_ticks_line_up_with_hits(pdhd):
    """On every plane the hit peaks trail the TP peaks on the same channel by a
    constant few ticks, the same early and late in the drift: the DTS -> tick
    scale is right (62.5 MHz / 32 instead of / 31.25 would open a ~70-tick gap
    across the window)."""
    ev = pdhd[3]
    tr, h = ev.trigger(), ev.hits()
    order = np.lexsort((tr.tp_tick_peak, tr.tp_channel))
    key = tr.tp_channel[order].astype(np.int64) * 1e6 + tr.tp_tick_peak[order]
    hk = h.channel.astype(np.int64) * 1e6 + h.tick
    j = np.clip(np.searchsorted(key, hk), 1, len(key) - 1)
    d = np.where(np.abs(key[j] - hk) < np.abs(key[j - 1] - hk), hk - key[j], hk - key[j - 1])
    ok = np.abs(d) < 20
    for view in (0, 1, 2):
        early = d[ok & (h.chan_view == view) & (h.tick < 2500)]
        late = d[ok & (h.chan_view == view) & (h.tick >= 2500)]
        assert len(early) > 500 and len(late) > 500
        assert -5 < np.median(early) < 15
        assert abs(np.median(early) - np.median(late)) < 1.5
