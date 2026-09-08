"""pylar - Pure-Python reader and data model for art-ROOT (LArSoft) event files.

Reads art-ROOT files directly with uproot (no ROOT, art or LArSoft needed).
Exposes hits, tracks, showers, space points, optical activity, and truth
with physical detector coordinates attached.

    from pylar import EventFile

    f = EventFile("reco.root")
    ev = f[0]
    hits = ev.hits()
    print(hits.w, hits.x, hits.integral)

Visualization is provided separately by the `pylarevd` package.
"""

__version__ = "0.1.0"

_EXPORTS = {
    "ArtFile": "artio", "ArtReadError": "artio", "UNKNOWN_EVENT_ID": "artio",
    "Event": "event", "EventFile": "event", "Hits": "event",
    "MCParticles": "event", "Neutrino": "event", "OpticalActivity": "event",
    "PrimaryInteraction": "event", "Showers": "event", "SpacePoints": "event",
    "Tracks": "event", "TruthDeposits": "event", "Vertices": "event",
    "Geometry": "geometry", "GeometryError": "geometry", "TPCBox": "geometry",
    "Containment": "geometry",
}

__all__ = sorted(list(_EXPORTS) + ["physics"])


def __getattr__(name: str):
    from importlib import import_module

    if name == "physics":
        return import_module(".physics", __name__)
    if name in _EXPORTS:
        return getattr(import_module("." + _EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
