"""Which exported geometry a file asks for, from its recorded Geometry service."""
from types import SimpleNamespace

from pylario.event import EventFile


def _recorded(cfg):
    stub = SimpleNamespace(art=SimpleNamespace(geometry_config=lambda: cfg))
    return EventFile._recorded_detector(stub)


def test_name_wins_over_gdml():
    assert _recorded({"Name": "dune10kt_v6_1x2x6", "GDML": "dune10kt_v6_refactored_1x2x6.gdml"}) \
        == "dune10kt_v6_1x2x6"


def test_gdml_stem_when_no_name():
    # the vertical-drift 1x8x14 files record only the GDML; LArSoft then names the detector after it
    assert _recorded({"GDML": "dunevd10kt_3view_30deg_v5_refactored_1x8x14ref.gdml"}) \
        == "dunevd10kt_3view_30deg_v5_refactored_1x8x14ref"


def test_nothing_recorded():
    assert _recorded({}) is None
