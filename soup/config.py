"""Run configuration: defaults <- JSON file <- CLI overrides."""
import argparse
import dataclasses
import json
from dataclasses import dataclass


@dataclass
class Config:
    mode: str = "ring"            # "ring", "bff" or "blocks"

    # ring mode
    M: int = 65536                # ring size in bytes
    R: int = 128                  # ip/head confinement radius around p

    # bff and blocks modes
    N: int = 1024                 # number of 64-byte tapes / blocks
    d: int = 2                    # blocks mode: partner drawn from +/-d blocks

    # reproduce cubff exactly: "none", "cubff" (heads read from tape[0..1],
    # pc=2) or "cubff_noheads" (heads at 0, pc=0 -- what our spec says).
    # Either value replaces our RNG, pairing and mutation with cubff's, so a
    # run can be diffed byte-for-byte against a cubff checkpoint.
    compat: str = "none"

    # language variants
    no_copy: int = 0              # '.' and ',' become no-ops
    indel: int = 0                # add insertions and deletions to mutation

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
    tau_epochs: int = 250         # epochs a window must stay abundant to persist
    null_ratio: float = 5.0       # and this many times its i.i.d. expectation
    compressor: str = "brotli6"   # brotli6 | brotli2 | brotli11 | zlib | zstd
    tau: int = 5                  # legacy: the old snapshot-counted tau, kept
                                  # only so pre-existing config.json files load
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

    @property
    def region_len(self):
        return 2 * self.R + 1 if self.mode == "ring" else 128

    @property
    def walk_len(self):
        """Steps a run takes if it only ever executes no-ops.

        The floor that `mean_steps` sits on when nothing loops.
        """
        if self.mode == "ring":
            return self.R + 1
        return 126 if self.compat == "cubff" else 128

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
    if cfg.mode not in ("ring", "bff", "blocks"):
        raise SystemExit("mode must be 'ring', 'bff' or 'blocks'")
    if cfg.compat not in ("none", "cubff", "cubff_noheads"):
        raise SystemExit("compat must be 'none', 'cubff' or 'cubff_noheads'")
    if cfg.compat != "none" and cfg.mode != "bff":
        raise SystemExit("compat modes only apply to --mode bff")
    if cfg.compressor not in ("brotli6", "brotli2", "brotli11", "zlib", "zstd"):
        raise SystemExit("unknown compressor: %s" % cfg.compressor)
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
        if cfg.mode == "blocks" and not 1 <= cfg.d <= cfg.N // 2:
            raise SystemExit("d must be between 1 and N/2")
    if not 0.0 <= cfg.mu <= 1.0:
        raise SystemExit("mu must be in [0, 1]")
    if cfg.k < 1:
        raise SystemExit("k must be >= 1")
    if cfg.window_size > 8:
        raise SystemExit("window_size must be <= 8 (keys are packed into uint64)")
