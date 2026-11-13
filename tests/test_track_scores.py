"""Pandora's TrackScore sentinel: -1 means "not characterised", not "shower"."""
import os

import numpy as np
import pytest

from pylario import EventFile

_DATA = os.environ.get("PYLARIO_TEST_DATA", os.environ.get("PYLAREVD_TEST_DATA", ""))
ATMNU = os.environ.get("PYLARIO_ATMNU", os.environ.get(
    "PYLAREVD_ATMNU", os.path.join(_DATA, "atmnu_radio_reco.root") if _DATA else ""))
needs_atmnu = pytest.mark.skipif(not os.path.exists(ATMNU), reason="neutrino sample absent")


@needs_atmnu
def test_uncharacterised_pfparticles_have_no_score_and_keep_both_fits():
    ef = EventFile(ATMNU)
    checked = 0
    for entry in range(min(len(ef), 4)):
        ev = ef[entry]
        scores = ev.pfp_track_scores()
        if scores is None:
            continue
        finite = scores[np.isfinite(scores)]
        assert (finite >= 0).all() and (finite <= 1).all(), "a sentinel leaked through as a score"
        sel = ev.pandora_best_match()
        for key, cls_prefer in (("tracks", "pandoraTrack"), ("showers", "pandoraShower")):
            if key not in sel:
                continue
            owner = ev._pfp_of({"tracks": "recob::Track", "showers": "recob::Shower"}[key], (cls_prefer,))
            unscored = (owner >= 0) & (owner < len(scores))
            unscored[unscored] = ~np.isfinite(scores[owner[unscored]])
            assert sel[key][unscored].all(), f"an unscored PFParticle lost its {key[:-1]} fit"
        checked += 1
    assert checked
