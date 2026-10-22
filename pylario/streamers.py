"""Sequential ROOT deserialiser for variable-length members.

(The columns that dominate truth decoding -- ``simb::MCTrajectory``, primitive
containers such as ``set<int>`` daughters, strings -- have vectorised readers
below that python-walk only the counts; they verify every header and hand the
column back to the generic walk on anything unexpected.)

:mod:`pylario.artio` decodes member-wise vectors as numpy slices, which is
fast but requires every member to have a fixed on-disk size.  That covers hits
and space points; it does not cover the products that carry containers --
``recob::Track`` (a trajectory of N points), ``recob::Shower``,
``recob::PFParticle``, ``simb::MCParticle``, ``recob::Wire``.

Those still arrive member-wise, so the columnar structure holds: what changes is
that a variable member's column cannot be strided over.  Empirically (verified
byte-exactly against dunesw output) such a column is written as::

    byte count (4) | version (2) | [count (4)][elements] x N objects

i.e. one header for the whole column, then a length-prefixed run per object.
This module walks that layout with a cursor instead of slicing it.

Nested objects carry their own byte-count prefix, which is read rather than
predicted -- so unlike the fast path this code never has to guess whether a
class uses a 6- or 10-byte header.
"""

from __future__ import annotations

import re
import struct
import weakref
from functools import lru_cache
from dataclasses import dataclass

import numpy as np

# C++ spelling -> (numpy dtype, on-disk size). ROOT writes big-endian.
_PRIMITIVES: dict[str, tuple[str, int]] = {
    "bool": (">?", 1), "char": (">i1", 1), "signed char": (">i1", 1),
    "unsigned char": (">u1", 1), "short": (">i2", 2), "unsigned short": (">u2", 2),
    "int": (">i4", 4), "unsigned int": (">u4", 4),
    "long": (">i8", 8), "unsigned long": (">u8", 8),
    "long long": (">i8", 8), "unsigned long long": (">u8", 8),
    "Long64_t": (">i8", 8), "ULong64_t": (">u8", 8),
    "float": (">f4", 4), "double": (">f8", 8),
    "Double32_t": (">f4", 4),      # stored as a float unless a range is declared
    "Float16_t": (">f2", 2),
    "size_t": (">u8", 8),
}

_BASE, _OBJECT, _ANY, _TOBJECT_BASE, _STL = 0, 61, 62, 66, 500
_OBJ_HEADER = 6
_TOBJECT_SIZE = 10             # version (2) + fUniqueID (4) + fBits (4)
_BYTECOUNT_MASK = 0x40000000


class StreamerError(RuntimeError):
    """Raised when a byte stream does not match the streamer description."""


#: Key added to a decoded object when some member could not be modelled.
UNPARSED = "__unparsed__"

# Caches, scoped PER FILE. Keying them by class name alone would let one file's
# schema decide how another file's identically-named class is read -- and since
# a partial memo is never retried, that silently loses members. A weak map also
# means closing a file drops its caches.
_FILE_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _caches(file) -> dict:
    try:
        entry = _FILE_CACHE.get(file)
    except TypeError:                       # not weak-referenceable
        entry = None
    if entry is None:
        entry = {"partial": {}, "why": {}, "elements": {}}
        try:
            _FILE_CACHE[file] = entry
        except TypeError:
            pass
    return entry


def reset_caches(file=None) -> None:
    """Forget cached streamer info and partial-parse decisions."""
    if file is None:
        _FILE_CACHE.clear()
    else:
        _FILE_CACHE.pop(file, None)


class Cursor:
    """A position in a buffer, with primitive readers."""

    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes, pos: int = 0):
        self.buf = buf
        self.pos = pos

    def skip(self, n: int) -> None:
        self.pos += n

    def u1(self) -> int:
        v = self.buf[self.pos]
        self.pos += 1
        return v

    def u2(self) -> int:
        v = struct.unpack_from(">H", self.buf, self.pos)[0]
        self.pos += 2
        return v

    def u4(self) -> int:
        v = struct.unpack_from(">I", self.buf, self.pos)[0]
        self.pos += 4
        return v

    #: numpy big-endian dtype -> struct format, for the one-value fast path.
    _STRUCT = {">?": "?", ">i1": "b", ">u1": "B", ">i2": "h", ">u2": "H",
               ">i4": "i", ">u4": "I", ">i8": "q", ">u8": "Q",
               ">f4": "f", ">f8": "d"}

    def scalar(self, dtype: str, size: int):
        """One value, without building a length-1 ndarray.

        The sequential reader does this 720k times for a single MCParticle
        product; np.frombuffer + astype + newbyteorder costs 1.9 us against
        0.2 us for struct.unpack_from.
        """
        fmt = self._STRUCT.get(dtype)
        if fmt is None:                      # e.g. float16, no struct code
            return self.array(dtype, 1, size)[0]
        v = struct.unpack_from(">" + fmt, self.buf, self.pos)[0]
        self.pos += size
        return v

    def array(self, dtype: str, count: int, size: int) -> np.ndarray:
        a = np.frombuffer(self.buf, dtype, count, self.pos)
        self.pos += count * size
        return a.astype(np.dtype(dtype).newbyteorder("="), copy=True)

    def header(self) -> tuple[int, int]:
        """Read a (byte count, version) prefix; returns (end position, version)."""
        raw = self.u4()
        version = self.u2()
        if raw & _BYTECOUNT_MASK:
            return self.pos - _OBJ_HEADER + 4 + (raw & ~_BYTECOUNT_MASK), version
        return -1, version          # no byte count available

    def string(self) -> str:
        """ROOT's length-prefixed string: one byte, or 255 then four.

        A length longer than the buffer is a misread of a wrong framing, not a
        string. Slicing tolerates it silently and still advances the cursor by
        the bogus length, which is how a mis-framed string desynchronised every
        member after it instead of raising and letting the caller retry.
        """
        n = self.u1()
        if n == 255:
            n = self.u4()
        if n > len(self.buf) - self.pos:
            raise StreamerError(
                f"string of {n} bytes with only {len(self.buf) - self.pos} left")
        s = self.buf[self.pos:self.pos + n]
        self.pos += n
        return s.decode("utf-8", "replace")


# ---- type-name parsing -----------------------------------------------------

def _split_template(arg: str) -> list[str]:
    """Split a template argument list at top-level commas."""
    out, depth, cur = [], 0, ""
    for ch in arg:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


@lru_cache(maxsize=512)
def parse_type(typename: str) -> tuple[str, list[str]]:
    """``vector<pair<int,float> >`` -> ``("vector", ["pair<int,float>"])``.

    Cached: pure, and called ~104k times per event for about six distinct
    type names, which cost ~2.4 s of an 8.2 s MCParticle decode.
    """
    t = typename.strip().rstrip("*").strip()
    m = re.match(r"^([A-Za-z_][\w:]*)\s*<(.*)>$", t, re.S)
    if not m:
        return t, ()
    return m.group(1), tuple(_split_template(m.group(2)))


_SEQUENCES = {"vector", "list", "deque", "set", "multiset", "unordered_set"}


# ---- readers ---------------------------------------------------------------

def read_value(file, typename: str, cur: Cursor, *, as_element: bool = False):
    """Read one value of *typename* at the cursor.

    ``as_element`` says this value is an ELEMENT of an enclosing container
    rather than a member of a class. The distinction matters for nested STL:
    ``vector<double>`` written as a class member carries its own (byte count,
    version) header, but the same type as an element of
    ``vector<vector<double>>`` carries only a count. Assuming a header in the
    element case broke recob::PCAxis; assuming none in the member case wiped
    out every MCParticle trajectory.
    """
    base = typename.strip()
    if base in _PRIMITIVES:
        dt, size = _PRIMITIVES[base]
        return cur.scalar(dt, size)
    if base in ("string", "std::string", "TString"):
        return _read_string(cur)

    kind, args = parse_type(base)
    kind = kind.replace("std::", "")
    if kind in _SEQUENCES:
        return read_sequence(file, args[0], cur, headered=not as_element)
    if kind == "pair":
        return (read_bare(file, args[0], cur), read_bare(file, args[1], cur))
    if kind == "map":
        return read_sequence(file, f"pair<{args[0]},{args[1]}>", cur, headered=True)
    return read_object(file, base, cur)


def _read_string(cur: Cursor) -> str:
    """A std::string that may or may not carry a (byte count, version) header.

    Read object-wise, a std::string member is wrapped like any other class;
    inside a member-wise column it is bare. Nothing in the streamer info says
    which, so try the header first and fall back -- the same trial the reader
    already applies to every other class.
    """
    save = cur.pos
    try:
        end, _version = cur.header()
        out = _read_string_bare(cur)
        if cur.pos == end:
            return out
    except (StreamerError, struct.error, ValueError, IndexError):
        pass
    cur.pos = save
    return _read_string_bare(cur)


def _read_string_bare(cur: Cursor) -> str:
    return cur.string()


def read_bare(file, typename: str, cur: Cursor):
    """Read a value with no object prefix (as inside a std::pair)."""
    base = typename.strip()
    if base in _PRIMITIVES:
        dt, size = _PRIMITIVES[base]
        return cur.scalar(dt, size)
    if base in ("string", "std::string", "TString"):
        return cur.string()
    kind, _args = parse_type(base)
    if kind.replace("std::", "") in _SEQUENCES | {"pair", "map"}:
        return read_value(file, base, cur)
    # Class members of a pair carry their own prefix in every case seen so far,
    # but fall back to a bare read rather than assume it.
    save = cur.pos
    try:
        return read_object(file, base, cur)
    except (StreamerError, struct.error, ValueError, IndexError):
        cur.pos = save
        return read_object(file, base, cur, headered=False)


def _as_pair(o):
    """Normalise a decoded std::pair to a 2-tuple.

    The element-wise path yields a tuple already; the member-wise path yields a
    dict whose halves are either plain ``first``/``second`` keys (POD members)
    or flattened ``first.<member>``/``second.<member>`` ones (object members).
    Callers should not have to know which layout the file used.
    """
    if not isinstance(o, dict):
        return o
    first, second = o.get("first"), o.get("second")
    a, b = {}, {}
    for key, value in o.items():
        if key.startswith("first."):
            a[key[len("first."):]] = value
        elif key.startswith("second."):
            b[key[len("second."):]] = value
    return (a or first, b or second)


def read_collection_header(cur: Cursor) -> tuple[int, bool, int]:
    """Read a container prefix: (element count, value class is foreign, end).

    When the member-wise bit is set the value class's version follows, and a
    version of 0 means that class is foreign, so a 4-byte checksum stands in
    for it.

    ``end`` is where the container's byte count says it finishes.  That is the
    only reliable way to tell whether its elements carry per-element headers:
    the foreign flag does NOT decide it (``ROOT::Math::PositionVector3D``
    elements are headered, ``lar::sparse_vector::datarange_t`` elements are
    not, and both report version 0).
    """
    end, version = cur.header()
    foreign = False
    if version & 0x4000:
        value_version = cur.u2()
        if value_version == 0:
            cur.skip(4)         # checksum in place of a version
            foreign = True
    return cur.u4(), foreign, end


def read_sequence(file, elem_type: str, cur: Cursor, *, headered: bool):
    """Read a length-prefixed run of *elem_type*.

    ``headered`` says whether the container carries its own prefix.  Inside a
    member-wise column it does not -- the column has a single shared prefix and
    each object contributes only its count.
    """
    end = -1
    if headered:
        n, _foreign, end = read_collection_header(cur)
    else:
        n = cur.u4()
    et = elem_type.strip()
    # Every element costs at least one byte, so a count larger than the bytes
    # remaining cannot be real -- it is a misread of a wrong framing guess.
    # Rejecting it here matters: without the check the reader faithfully parses
    # the phantom elements until the buffer runs out, which turned one
    # simb::MCTruth (15 trajectories) into 2.35 million TLorentzVector reads and
    # never finished.
    budget = (end if end >= 0 else len(cur.buf)) - cur.pos
    if n < 0 or n > max(budget, 0):
        raise StreamerError(
            f"vector<{et}>: element count {n} exceeds the {budget} bytes left")
    if et in _PRIMITIVES:                       # fast path: contiguous PODs
        dt, size = _PRIMITIVES[et]
        return cur.array(dt, n, size)

    if n == 0:
        return []
    kind = parse_type(et)[0].replace("std::", "")
    composite = kind in _SEQUENCES | {"pair", "map"}

    # How class elements are framed is not derivable from streamer info, so try
    # each layout and keep the one that lands exactly on the container's end.
    # Member-wise (column per member) first: that is what ROOT writes when the
    # member-wise bit is set, and it is the only reading that recovers e.g.
    # multi-ROI wire waveforms.
    start = cur.pos
    # "memberwise" = one column per member (what ROOT writes for the member-wise
    # bit, including for std::pair elements, whose two halves become two
    # columns); "elementwise" = each element read whole, in order.
    attempts = [("memberwise", "mw"), ("elementwise", "ew")] if composite else \
               [("memberwise", "mw"), ("headerless", False), ("headered", True)]
    last_error = None
    for label, how in attempts:
        cur.pos = start
        try:
            if how == "mw":
                out = read_memberwise_elements(file, et, cur, n)
            elif how == "ew":
                out = [read_value(file, et, cur, as_element=True)
                       for _ in range(n)]
            else:
                out = [read_object(file, et, cur, headered=how) for _ in range(n)]
        except (StreamerError, struct.error, ValueError, IndexError) as exc:
            last_error = exc
            continue
        if end < 0 or cur.pos == end:
            # A pair reaches us as {"first","second"} from the member-wise path
            # and as a tuple from the element-wise one; normalise so callers
            # never have to care which layout the file happened to use.
            if kind == "pair":
                out = [_as_pair(o) for o in out]
            return out
        last_error = StreamerError(
            f"{et} as {label}: ended at {cur.pos}, container ends at {end}")
    raise StreamerError(f"vector<{et}>: cannot parse ({last_error})")


def read_memberwise_elements(file, classname: str, cur: Cursor, n: int) -> list:
    """Read *n* objects of *classname* stored member-wise: one column per member.

    This is ROOT's real member-wise layout, applied recursively -- the same rule
    the top-level product vector uses.  A base class contributes its own columns
    inline; a nested-object member's column holds *n* objects written
    object-wise, each with its own prefix (which is why ``geo::WireID`` is 41
    bytes inside ``recob::Hit``, and why a ``PositionVector3D`` column holds
    22-byte ``Cartesian3D`` objects).
    """
    columns: dict = {}
    for el in _elements(file, classname):
        ftype = el.ftype
        name = el.name
        if ftype == _TOBJECT_BASE or (ftype == _BASE and name == "TObject"):
            cur.skip(_TOBJECT_SIZE * n)
        elif ftype == _BASE:
            base_rows = read_memberwise_elements(file, name, cur, n)
            for key in (base_rows[0].keys() if base_rows else ()):
                columns[key] = [row[key] for row in base_rows]
        elif ftype == _STL:
            columns[name] = read_memberwise_column(file, el, cur, n)
        elif ftype in (_OBJECT, _ANY):
            # Flatten nested members to "outer.inner" exactly as read_object
            # does, so a caller sees the same keys whichever layout the file
            # used.
            objs = [read_object(file, el.typename, cur) for _ in range(n)]
            for key in (objs[0].keys() if objs else ()):
                columns[f"{name}.{key}"] = [o[key] for o in objs]
        else:
            from .artio import _ROOT_TYPES
            if ftype not in _ROOT_TYPES:
                raise StreamerError(f"{name}: unsupported fType={ftype}")
            dt, size = _ROOT_TYPES[ftype]
            length = el.array_length
            if length <= 1:
                columns[name] = cur.array(dt, n, size)
            else:
                columns[name] = cur.array(dt, n * length, size).reshape(n, length)
    return [{k: v[i] for k, v in columns.items()} for i in range(n)]


def read_object(file, classname: str, cur: Cursor, *, headered: bool = True) -> dict:
    """Read one object of *classname*, returning a flat dict of members.

    The object's byte count is read from the stream and used to check that
    parsing consumed exactly the right number of bytes, so a layout mistake
    surfaces here rather than as corrupt numbers downstream.

    Whether a by-value member carries its own prefix is not recorded in
    streamer info, so both readings are tried and the one that fits the byte
    count is kept.  If neither does, the byte count still says exactly where
    the object ends: we keep the members read up to the first unmodelled one,
    resynchronise, and flag the shortfall under :data:`UNPARSED`.  Losing a
    flag word must not cost us a whole trajectory.
    """
    end = -1
    if headered:
        end, version = cur.header()
        # Version 0 means the class is foreign: a 4-byte checksum stands in for
        # the version. Read that from the stream rather than predicting it.
        if version == 0:
            cur.skip(4)

    body = cur.pos
    elements = _elements(file, classname)

    memo = _caches(file)

    # Always attempt the FULL parse, for every object.
    #
    # This used to be skipped once a partial-parse limit had been memoised for
    # the class -- which made decoding order-dependent and silently lossy: the
    # same class can be framed differently at different call sites (simb::MCParticle
    # parses whole inside MCTruth::fPartList but not inside MCNeutrino::fNu), so
    # one failure taught the reader to truncate *every* later object of that
    # class. Visiting one event then cost every subsequent event its truth.
    # The memo below now short-circuits only the expensive probe, never the
    # parse itself, so a readable object is always read in full.
    last_error = None
    for nested_headered in (True, False):
        cur.pos = body
        out: dict = {}
        try:
            for el in elements:
                _read_element(file, el, cur, out, end, nested_headered)
        except (StreamerError, struct.error, ValueError, IndexError) as exc:
            last_error = exc
            continue
        # Require the byte count to be consumed EXACTLY. Accepting
        # cur.pos <= end silently tolerates a truncated parse, which is how
        # whole members (recob::Wire's waveforms) went missing while every
        # object still looked like it decoded cleanly.
        if end < 0 or cur.pos == end:
            return out
        last_error = StreamerError(
            f"{classname}: consumed {cur.pos - body} bytes, "
            f"byte count says {end - body}")
        if end < 0:
            break
    if end < 0:
        raise StreamerError(f"{classname}: cannot parse ({last_error})")

    # This object genuinely does not parse. Fall back to reading the leading
    # members and resynchronising on the byte count. How many are readable is a
    # property of the class, so that much is worth remembering.
    # Keyed by (class, object size), NOT by class alone. How far a class can be
    # read depends on the context it was written in, not just on the class: the
    # same simb::MCParticle recovered 6 members inside one MCTruth product and
    # 11 inside another, so a class-only memo let whichever was decoded first
    # truncate the other. Object size distinguishes the contexts cheaply and
    # cannot collide across them.
    key = (classname, end - body)
    limit = memo["partial"].get(key)
    if limit is None:
        limit = _probe_limit(file, classname, elements, cur.buf, body, end)
        memo["partial"][key] = limit
        memo["why"][key] = str(last_error)

    out = {}
    cur.pos = body
    try:
        for el in elements[:limit]:
            _read_element(file, el, cur, out, end)
    except (StreamerError, struct.error, ValueError, IndexError):
        out = {}
    cur.pos = end
    out[UNPARSED] = memo["why"].get(key, "unmodelled member")
    return out


def _probe_limit(file, classname: str, elements, buf: bytes, body: int,
                 end: int) -> int:
    """How many leading members of *classname* can be read before one fails."""
    probe = Cursor(buf, body)
    n = 0
    for el in elements:
        mark = probe.pos
        try:
            _read_element(file, el, probe, {}, end)
            if probe.pos > end:
                raise StreamerError("overran")
        except Exception:
            probe.pos = mark
            break
        n += 1
    return n


def _is_checksummed(file, classname: str) -> bool:
    try:
        info = file.streamer_named(classname)
    except Exception:
        return False
    return info is not None and int(info.class_version) == 1


@dataclass(frozen=True)
class ElementSpec:
    """The streamer facts we actually use, extracted once per class.

    Reading them off uproot's model object instead costs a ``member()`` lookup
    per element *per object*: that walk dominated the decode of nested products
    (``simb::MCTruth`` did not finish in ten minutes; it takes well under a
    second here).
    """

    name: str
    typename: str
    ftype: int
    array_length: int

    @classmethod
    def of(cls, el) -> "ElementSpec":
        """Adopt a raw uproot streamer element (no-op if already a spec)."""
        if isinstance(el, cls):
            return el
        return cls(name=el.name, typename=el.typename,
                   ftype=int(el.member("fType")),
                   array_length=int(el.member("fArrayLength") or 0))


def _elements(file, classname: str):
    """Streamer elements for *classname*, memoised per file.

    Without the memo this is re-resolved on every object at every nesting
    depth -- 125k lookups to decode eleven tracks, most of the decode time.
    """
    cache = _caches(file)["elements"]
    hit = cache.get(classname)
    if hit is None:
        info = file.streamer_named(classname)
        if info is None:
            raise StreamerError(f"no streamer info for {classname!r}")
        hit = cache[classname] = [ElementSpec.of(el) for el in info.elements]
    return hit


def _read_nested(file, typename: str, cur: Cursor, budget_end: int,
                 prefer_headered: bool = True) -> dict:
    """Read a nested object whose header may or may not be present.

    Whether ROOT writes a prefix for a by-value member depends on the member's
    type in ways streamer info does not record, and the obvious test -- does the
    next word look like a byte count? -- is useless here, because a plausible
    coordinate such as 36.85 has the byte-count bit set in its float encoding.

    So try it with a header and fall back without one, using the *enclosing*
    object's byte count to tell the two apart. That is authoritative: a wrong
    guess overruns a known boundary.
    """
    save = cur.pos
    try:
        sub = read_object(file, typename, cur, headered=prefer_headered)
        if budget_end >= 0 and cur.pos > budget_end:
            raise StreamerError("overran the enclosing object")
        return sub
    except (StreamerError, struct.error, ValueError, IndexError):
        cur.pos = save
        return read_object(file, typename, cur, headered=not prefer_headered)


def _read_element(file, el, cur: Cursor, out: dict, budget_end: int = -1,
                  nested_headered: bool = True) -> None:
    ftype = el.ftype
    name = el.name
    typename = el.typename

    if ftype == _TOBJECT_BASE:
        cur.skip(_TOBJECT_SIZE)
        return
    if ftype == _BASE:
        if name in ("TObject",):
            cur.skip(_TOBJECT_SIZE)
            return
        out.update(_read_nested(file, name, cur, budget_end, nested_headered))
        return
    if ftype == _STL:
        out[name] = read_value(file, typename, cur)
        return
    if ftype in (_OBJECT, _ANY):
        for k, v in _read_nested(file, typename, cur, budget_end,
                                 nested_headered).items():
            out[f"{name}.{k}"] = v
        return

    # plain data, possibly a fixed-size array
    from .artio import _ROOT_TYPES
    if ftype not in _ROOT_TYPES:
        raise StreamerError(f"{name}: unsupported streamer fType={ftype} ({typename})")
    dt, size = _ROOT_TYPES[ftype]
    length = el.array_length
    if length <= 1:
        out[name] = cur.scalar(dt, size)
    else:
        out[name] = cur.array(dt, length, size)


#: One serialised TLorentzVector: byte count + version, TObject (version,
#: fUniqueID, fBits), the TVector3 fP with its own byte count + version and
#: TObject, fX fY fZ, then fE. 64 bytes; the byte counts say 60 and 36.
_TLV_REC = np.dtype([("bc", ">u4"), ("ver", ">u2"), ("obj", "V10"),
                     ("vbc", ">u4"), ("vver", ">u2"), ("vobj", "V10"),
                     ("x", ">f8"), ("y", ">f8"), ("z", ">f8"), ("e", ">f8")])
_TLV_BC, _TV3_BC = 60, 36
#: simb::MCTrajectory's two members, by on-disk type (spaces dropped)
_TRAJ_STEPS = "vector<pair<TLorentzVector,TLorentzVector>>"
_TRAJ_PROC = "vector<pair<unsignedlong,unsignedchar>>"
_HEAD = struct.Struct(">IH")             # byte count, version
_MW_HEAD = struct.Struct(">IHHII")       # byte count, version, value version, checksum, count
_TRAJ_HEAD = struct.Struct(">IHIHHII")   # an object's (byte count, version), then _MW_HEAD
_U4 = struct.Struct(">I")


def _pieces(arr: np.ndarray, counts: np.ndarray) -> list:
    """*arr* cut into consecutive views of the given lengths. np.split does the
    same at ~3 us a piece: 2.5 s for one event's 761k particles."""
    ends = np.cumsum(counts).tolist()
    return [arr[a:b] for a, b in zip([0] + ends[:-1], ends)]


def _string_column(cur: Cursor, n: int, end: int) -> list:
    """*n* consecutive bare strings, in one tight loop (Cursor.string per value
    cost 1.8 us; 1.5M of them in one MCParticle product)."""
    buf, pos = cur.buf, cur.pos
    limit = end if end >= 0 else len(buf)
    u4 = _U4.unpack_from
    out = []
    append = out.append
    for _ in range(n):
        k = buf[pos]
        pos += 1
        if k == 255:
            k = u4(buf, pos)[0]
            pos += 4
        append(buf[pos:pos + k].decode("utf-8", "replace"))
        pos += k
        if pos > limit:
            raise StreamerError(f"string column runs {pos - limit} bytes past its end")
    cur.pos = pos
    return out


def _is_trajectory_class(file, classname: str) -> bool:
    """Does *classname* have exactly simb::MCTrajectory's on-disk shape?"""
    try:
        els = _elements(file, classname)
    except StreamerError:
        return False
    types = [e.typename.replace(" ", "") for e in els]
    return types == [_TRAJ_STEPS, _TRAJ_PROC] and all(e.ftype == _STL for e in els)


def _trajectory_column(file, classname: str, cur: Cursor, n: int):
    """A member-wise column of *n* simb::MCTrajectory objects, decoded at once.

    The generic reader walks this object by object, building a dict per
    TLorentzVector: 72 s for the 761k particles (1.5M steps) of one
    ProtoDUNE-HD event with cosmics. Here python reads only the counts; every
    TLorentzVector then comes from one numpy view. Layout, checked byte-exactly
    against the generic reader::

        per object: byte count | version
          steps:    byte count | version|0x4000 | value version 0 | checksum | N
                    N first TLorentzVectors, then N second ones (64 B each)
          process:  byte count | version|0x4000 | value version 0 | checksum | K
                    K unsigned long (step index), then K unsigned char (process)

    Returns ``[{steps_name: (N, 8) array [x y z t px py pz E], proc_name:
    [(index, process), ...]}, ...]``, the steps in the array form the split-file
    decoder already gives, or None if the bytes do not have exactly this
    layout (every header and byte count is checked), so the caller can fall
    back to the generic reader.
    """
    buf, pos = cur.buf, cur.pos
    steps_name, proc_name = (e.name for e in _elements(file, classname))
    obj_head, mw_head = _TRAJ_HEAD.unpack_from, _MW_HEAD.unpack_from
    mask, mw, low = _BYTECOUNT_MASK, 0x4000, ~_BYTECOUNT_MASK
    counts, starts = [], []
    procs: list = [[] for _ in range(n)]
    try:
        for i in range(n):
            bc, _ver, sbc, sver, svv, _ck, k = obj_head(buf, pos)
            end = pos + 4 + (bc & low)
            p = pos + 10 + (sbc & low)            # process vector
            # flags, member-wise bits, value version 0 (checksummed) and
            # the steps byte count matching 2k 64-byte records
            if not (bc & sbc & mask and sver & mw) or svv or pos + 22 + 128 * k != p:
                return None
            counts.append(k)
            starts.append(pos + 22)
            pbc, pver, pvv, _ck, m = mw_head(buf, p)
            if not (pbc & mask and pver & mw) or pvv or p + 16 + 9 * m != end \
                    or p + 4 + (pbc & low) != end:
                return None
            if m:
                idx = np.frombuffer(buf, ">u8", m, p + 16).astype(np.uint64)
                code = np.frombuffer(buf, "u1", m, p + 16 + 8 * m)
                procs[i] = list(zip(idx, code))
            pos = end
    except (struct.error, ValueError):
        return None
    if pos > len(buf):
        return None

    counts = np.asarray(counts, np.int64)
    starts = np.asarray(starts, np.int64)
    tot = int(counts.sum())
    if tot:
        # all records of the column, in object order: object i's 2N records
        # start at starts[i]; gather them with one fancy index over a 64-byte
        # strided view of the buffer
        raw = np.frombuffer(buf, np.uint8)
        rows = np.lib.stride_tricks.as_strided(
            raw, shape=(len(raw) - _TLV_REC.itemsize + 1, _TLV_REC.itemsize),
            strides=(1, 1), writeable=False)
        first = np.repeat(np.cumsum(counts) - counts, counts)       # per step
        k = np.arange(tot) - first                                 # step within object
        own = np.repeat(starts, counts)
        a = rows[own + 64 * k].view(_TLV_REC).reshape(tot)
        b = rows[own + 64 * (np.repeat(counts, counts) + k)].view(_TLV_REC).reshape(tot)
        for r in (a, b):
            if np.any(r["bc"] != mask | _TLV_BC) or np.any(r["vbc"] != mask | _TV3_BC):
                return None
        both = np.column_stack([a["x"], a["y"], a["z"], a["e"],
                                b["x"], b["y"], b["z"], b["e"]]).astype(np.float64)
    else:
        both = np.zeros((0, 8))
    steps = _pieces(both, counts)
    cur.pos = pos
    return [{steps_name: s, proc_name: pr} for s, pr in zip(steps, procs)]


def _primitive_runs(cur: Cursor, typename: str, n: int, end: int):
    """*n* length-prefixed runs of a primitive type (``set<int>`` daughters), as
    one numpy read split per object, instead of one small array per object
    (3.4 s for 761k particles). None, with the cursor untouched, if a count
    overruns the column."""
    dt, size = _PRIMITIVES[typename]
    buf, pos = cur.buf, cur.pos
    limit = end if end >= 0 else len(buf)
    u4 = _U4.unpack_from
    counts, starts = [], []
    try:
        for _ in range(n):
            c = u4(buf, pos)[0]
            counts.append(c)
            starts.append(pos + 4)
            pos += 4 + c * size
    except struct.error:
        return None
    if pos > limit:
        return None
    counts = np.asarray(counts, np.int64)
    tot = int(counts.sum())
    first = np.repeat(np.cumsum(counts) - counts, counts)
    at = np.repeat(np.asarray(starts, np.int64), counts) + size * (np.arange(tot) - first)
    raw = np.frombuffer(buf, np.uint8)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(max(len(raw) - size + 1, 0), size),
                                           strides=(1, 1), writeable=False)
    vals = rows[at].view(dt).reshape(tot).astype(np.dtype(dt).newbyteorder("="))
    cur.pos = pos
    return _pieces(vals, counts)


def read_memberwise_column(file, el, cur: Cursor, n: int) -> list:
    """Read one variable-length member for all *n* objects of a member-wise vector.

    The column carries a single (byte count, version) prefix; each object then
    contributes its own length-prefixed run.
    """
    if n == 0:
        return []
    el = ElementSpec.of(el)      # artio passes raw uproot elements
    ftype = el.ftype

    if ftype != _STL:
        # A nested object member is written object-wise within its column:
        # no shared prefix, but every object keeps its own byte-count header
        # (this is what makes geo::WireID 41 bytes inside recob::Hit).
        if _is_trajectory_class(file, el.typename):
            start = cur.pos
            fast = _trajectory_column(file, el.typename, cur, n)
            if fast is not None:
                return fast
            cur.pos = start
        return [read_object(file, el.typename, cur) for _ in range(n)]

    # An STL member's column has one shared prefix, then a length-prefixed run
    # per object with no per-object header. The prefix carries the value-class
    # version (and a checksum when that is 0) exactly as a standalone container
    # does -- skipping those misaligns every object after the first.
    end, version = cur.header()
    if version & 0x4000:
        if cur.u2() == 0:
            cur.skip(4)
    kind, args = parse_type(el.typename)
    kind = kind.replace("std::", "")
    if el.typename.strip() in ("string", "std::string"):
        values = _string_column(cur, n, end)
    elif kind in _SEQUENCES:
        values = None
        if args[0].strip() in _PRIMITIVES:
            values = _primitive_runs(cur, args[0].strip(), n, end)
        if values is None:          # not primitive, or let the generic reader raise
            values = [read_sequence(file, args[0], cur, headered=False)
                      for _ in range(n)]
    elif kind == "map":
        values = [read_sequence(file, f"pair<{args[0]},{args[1]}>", cur,
                                headered=False) for _ in range(n)]
    else:
        raise StreamerError(f"{el.name}: unsupported STL container {el.typename}")

    if end >= 0:
        if cur.pos > end:
            raise StreamerError(
                f"{el.name}: read {cur.pos - end} bytes past the end of the column")
        cur.pos = end
    return values
