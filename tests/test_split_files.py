"""Split (older-art) files: ProtoDUNE-SP Prod4 reco1, dunetpc v09.

Point PYLARIO_PDSP at such a file; the tests skip without it.
"""

import os

import numpy as np
import pytest

PDSP = os.environ.get("PYLARIO_PDSP", "")
needs_pdsp = pytest.mark.skipif(not os.path.exists(PDSP), reason="PDSP sample absent")


@pytest.fixture(scope="module")
def ev():
    from pylario import EventFile
    return EventFile(PDSP)[0]


@needs_pdsp
def test_event_ids_from_the_old_auxiliary_layout():
    from pylario import ArtFile
    ids = ArtFile(PDSP).event_ids()
    assert (ids["run"] > 0).all() and (np.diff(ids["event"]) == 1).all()


@needs_pdsp
def test_hits_match_the_channel_map(ev):
    h = ev.hits("hitpdune")
    assert len(h) > 1000 and h.n_bad_geometry == 0


@needs_pdsp
def test_track_trajectories_decode(ev):
    tr = ev.tracks(best_match=False)
    assert len(tr) > 0
    pts = np.concatenate([p for p in tr.points if len(p)])
    assert np.isfinite(pts).all() and np.abs(pts).max() < 2000


@needs_pdsp
def test_mc_trajectory_steps_are_on_shell(ev):
    raw = ev._mc_raw()[1]
    for i in range(0, len(raw["fpdgCode"]), 5000):
        st = raw["ftrajectory"][i]["ftrajectory"]
        if not len(st):
            continue
        p, e, m = st[:, 4:7], st[:, 7], raw["fmass"][i]
        assert np.allclose((p ** 2).sum(1) + m ** 2, e ** 2, rtol=1e-6, atol=1e-9)


@needs_pdsp
def test_beam_particle_without_its_halo(ev):
    inter = ev.true_interaction()
    assert inter is not None and len(inter.particles) == 1
    assert ev.capabilities()["truth"]


@needs_pdsp
def test_every_particle_has_its_generator(ev):
    # LArG4's MCParticle-MCTruth association (an Assns carrying data), with the
    # MCTruth ProductIDs resolved to their generator module labels
    mc = ev.mc_particles()
    assert mc.generator is not None and (mc.generator >= 0).all()
    names = set(mc.generators)
    assert {"cosmicgenerator", "generator", "ar39"} <= names
    # the beam interaction's particles all come from the beam generator
    fi = np.isin(mc.track_id, ev.neutrino_track_ids())
    assert fi.any()
    assert {mc.generators[g] for g in np.unique(mc.generator[fi])} == {"generator"}
