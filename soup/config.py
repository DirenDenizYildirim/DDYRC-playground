"""Run configuration: defaults <- JSON file <- CLI overrides."""
import argparse
import dataclasses
import json
from dataclasses import dataclass


@dataclass
class Config:
    mode: str = "ring"            # "ring" or "bff"

    # ring mode
    M: int = 65536                # ring size in bytes
    R: int = 128                  # ip/head confinement radius around p

    # bff mode
    N: int = 1024                 # number of 64-byte tapes

    # both modes
    k: int = 8192                 # step budget per run
    mu: float = 0.00024           # per-byte mutation probability per epoch
    seed: int = 0
    epochs: int = 50000

    # measurement
    snapshot_interval: int = 50   # epochs between metric rows
    tape_dump_interval: int = 1000  # epochs between raw .npy dumps
    window_size: int = 8          # window length for A(t)
    c_min: int = 8                # count threshold for "abundant"
    tau: int = 5                  # consecutive snapshots needed for "persistent"
    top_window: int = 16          # window length for the pattern log
    top_k: int = 10               # how many patterns to log
    kymo_width: int = 4096        # bytes of memory recorded per kymograph row
    zlib_level: int = 9

    # "wrap" (default): heads wrap modulo the region, so a head move is never
    # a termination condition.  "halt": a head move that would leave the region
    # ends the run instead.  Both are readings of "heads are confined"; the
    # default is the one the rest of this repo's results use.
    head_bound: str = "wrap"

    # stop early once a takeover is detected (0 = disabled; run all epochs).
    # The detector is the one in soup.analyze: high-order entropy at least
    # HOE_RISE above its epoch-0 value for SUSTAIN consecutive snapshots.
    # The value is how many further epochs to run after it fires, so the
    # post-transition plateau is still measured.
    stop_after_takeover: int = 0

    # validation helpers (off by default; they change the initial condition)
    plant: int = 0                # plant this many copies of the hand-written
                                  # replicator at init -- validation only

    @property
    def n_bytes(self):
        return self.M if self.mode == "ring" else self.N * 64

    @property
    def ticks_per_epoch(self):
        return self.M // 64 if self.mode == "ring" else self.N // 2

    def to_json(self):
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)


_TYPES = {f.name: f.type for f in dataclasses.fields(Config)}


def add_cli_args(parser):
    parser.add_argument("--config", help="JSON file with any of the fields below")
    for f in dataclasses.fields(Config):
        kind = {int: int, float: float, str: str}[
            f.type if not isinstance(f.type, str) else {"int": int, "float": float, "str": str}[f.type]
        ]
        parser.add_argument("--" + f.name.replace("_", "-"), dest=f.name,
                            type=kind, default=None,
                            help=f"(default {f.default})")
    return parser


def build_config(args):
    cfg = Config()
    if getattr(args, "config", None):
        with open(args.config) as fh:
            for key, value in json.load(fh).items():
                if key not in _TYPES:
                    raise SystemExit(f"unknown config key: {key}")
                setattr(cfg, key, value)
    for name in _TYPES:
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)
    validate(cfg)
    return cfg


def validate(cfg):
    if cfg.mode not in ("ring", "bff"):
        raise SystemExit("mode must be 'ring' or 'bff'")
    if cfg.head_bound not in ("wrap", "halt"):
        raise SystemExit("head_bound must be 'wrap' or 'halt'")
    if cfg.mode == "ring":
        if cfg.M < 2 * cfg.R + 1:
            raise SystemExit("M must be at least 2R+1")
        if cfg.M % 64:
            raise SystemExit("M must be a multiple of 64 (epoch = M/64 ticks)")
    else:
        if cfg.N < 2 or cfg.N % 2:
            raise SystemExit("N must be a positive even number")
    if not 0.0 <= cfg.mu <= 1.0:
        raise SystemExit("mu must be in [0, 1]")
    if cfg.k < 1:
        raise SystemExit("k must be >= 1")
    if cfg.window_size > 8:
        raise SystemExit("window_size must be <= 8 (keys are packed into uint64)")
