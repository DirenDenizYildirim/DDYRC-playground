"""A hand-written self-replicating program, used to validate the interpreter.

    \\x08 [ - }}}}}}}} ] ++++++++ [ . > } ]
    ^    ^-------------^ ^------^ ^-------^
    |    move head1 +64  restore   copy bytes from head0 to head1 until a
    |    (8 x 8), zeroing         zero byte at head0 is reached
    |    the counter
    counter byte, value 8 (not an instruction, so it is a no-op when executed)

Run with ip = head0 = head1 = 0 at the start of a region of zeros, it writes an
exact byte-for-byte copy of its own 25 bytes at offset +64 -- including the
counter, which the '++++++++' run restores before the copy loop reads it.  It
is therefore a true replicator and not just a program that copies its body.

It is not the product of evolution; nothing in the simulator knows about it.
It exists so the test-suite can assert that the interpreter really does let a
program copy itself, in both modes.
"""
import numpy as np

OFFSET = 64  # where the copy lands, relative to head0's starting position

PROGRAM = b"\x08[-" + b"}" * 8 + b"]" + b"+" * 8 + b"[.>}]"

assert len(PROGRAM) == 25
assert 0 not in PROGRAM  # the copy loop stops at the first zero byte


def as_array():
    return np.frombuffer(PROGRAM, dtype=np.uint8).copy()
