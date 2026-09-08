# pylario — pure-python reader and data model for art-ROOT (LArSoft) files

Reads **art-ROOT files directly** in pure Python. At read time there is no ROOT, no art, no LArSoft, and no compiled extension — just `uproot` + `numpy`.

Exposes reconstructed hits, tracks, space points, showers, vertices, optical activity, and truth with physical detector coordinates attached.

```
art-ROOT file ──uproot──► artio.py ──► event.py ──► Hits, Tracks, Neutrino, ...
                         (pure python, no LArSoft)
```

## Installation

```bash
pip install pylario
# or for remote XRootD / EOS support:
pip install "pylario[xrootd]"
```

## Quick Start

```python
from pylario import EventFile

# Open an art file (geometry is auto-detected from embedded FHiCL metadata)
f = EventFile("reco.root")
print(f"Events: {len(f)}")

ev = f[0]
print("Run/SubRun/Event:", ev.id)

# Hits with physical coordinates (wire coordinate w [cm], drift x [cm])
hits = ev.hits()
print(f"Hits: {len(hits)}")
print("w:", hits.w)
print("x:", hits.x)
print("charge:", hits.integral)

# Filter hits with boolean mask
bright = hits[hits.integral > 200]

# 3D Space points
sp = ev.spacepoints()
print("XYZ:", sp.xyz)

# Reconstructed tracks as 3D polylines
tracks = ev.tracks()
for trk in tracks.points:
    print("Track length:", len(trk))

# True interaction / Neutrino truth
nu = ev.neutrino()
if nu:
    print(nu.describe())
```

## Visualization

For 2D/3D event displays, interactive HTML, and browser application, install [`pylarevd`](https://github.com/pgranger23/pylarevd):

```bash
pip install pylarevd
```

```python
import pylarevd

pylarevd.display(ev).save("event0.png")
pylarevd.display_3d(ev).show()
```

## License

MIT License.
