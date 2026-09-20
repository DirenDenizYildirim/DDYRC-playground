"""The byte machine.

A single interpreter is shared by both modes.  It operates on a *region*: a
contiguous uint8 buffer.  Callers are responsible for materialising the region
(``bff`` concatenates two 64-byte tapes; ``ring`` copies a 2R+1 window out of
the ring and writes it back afterwards).

Instruction set (byte values of these ASCII characters).  Every other byte
value is a no-op that still costs one step.

    <  >   move head0 by -1 / +1
    {  }   move head1 by -1 / +1
    -  +   decrement / increment byte at head0 (wrapping mod 256)
    .      copy region[head0] -> region[head1]
    ,      copy region[head1] -> region[head0]
    [      if region[head0] == 0, jump forward to the matching ]
    ]      if region[head0] != 0, jump back to the matching [

Execution ends when the step budget k is exhausted, when the instruction
pointer runs off the region, or when a bracket has no match.

Head confinement: head0/head1 wrap modulo the region length.  They can never
leave the region, so a head move is never a termination condition -- this is
what "heads are confined to within +/-R of p" is taken to mean, and it keeps
the list of termination conditions exactly the three above.  Brackets are
matched by scanning the region *at execution time*, so self-modifying code
changes the control flow it is running under.
"""
import numpy as np
from numba import njit

# --- instruction byte values -------------------------------------------------
OP_H0_DEC = ord("<")   # 60
OP_H0_INC = ord(">")   # 62
OP_H1_DEC = ord("{")   # 123
OP_H1_INC = ord("}")   # 125
OP_DEC = ord("-")      # 45
OP_INC = ord("+")      # 43
OP_COPY_01 = ord(".")  # 46
OP_COPY_10 = ord(",")  # 44
OP_LOOP_BEG = ord("[") # 91
OP_LOOP_END = ord("]") # 93

INSTRUCTIONS = bytes(sorted([
    OP_H0_DEC, OP_H0_INC, OP_H1_DEC, OP_H1_INC, OP_DEC, OP_INC,
    OP_COPY_01, OP_COPY_10, OP_LOOP_BEG, OP_LOOP_END,
]))

# --- termination codes -------------------------------------------------------
TERM_BUDGET = 0      # step budget k exhausted
TERM_OFF_REGION = 1  # instruction pointer left the region
TERM_UNMATCHED = 2   # a bracket had no match

TERM_NAMES = {TERM_BUDGET: "budget", TERM_OFF_REGION: "off_region",
              TERM_UNMATCHED: "unmatched"}


@njit(cache=True, nogil=True)
def run_region(buf, ip, h0, h1, k):
    """Execute ``buf`` in place.  Returns (steps, n_copies, term_code).

    ``n_copies`` counts executed ``.`` and ``,`` instructions.
    """
    n = buf.shape[0]
    steps = 0
    ncopy = 0
    while steps < k:
        if ip < 0 or ip >= n:
            return steps, ncopy, TERM_OFF_REGION
        c = buf[ip]
        if c == 62:        # '>'
            h0 += 1
            if h0 >= n:
                h0 -= n
        elif c == 60:      # '<'
            h0 -= 1
            if h0 < 0:
                h0 += n
        elif c == 125:     # '}'
            h1 += 1
            if h1 >= n:
                h1 -= n
        elif c == 123:     # '{'
            h1 -= 1
            if h1 < 0:
                h1 += n
        elif c == 43:      # '+'
            buf[h0] = (buf[h0] + 1) & 255
        elif c == 45:      # '-'
            buf[h0] = (buf[h0] + 255) & 255
        elif c == 46:      # '.'
            buf[h1] = buf[h0]
            ncopy += 1
        elif c == 44:      # ','
            buf[h0] = buf[h1]
            ncopy += 1
        elif c == 91:      # '['
            if buf[h0] == 0:
                depth = 1
                j = ip + 1
                while j < n:
                    cj = buf[j]
                    if cj == 91:
                        depth += 1
                    elif cj == 93:
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                if j >= n:
                    return steps + 1, ncopy, TERM_UNMATCHED
                ip = j
        elif c == 93:      # ']'
            if buf[h0] != 0:
                depth = 1
                j = ip - 1
                while j >= 0:
                    cj = buf[j]
                    if cj == 93:
                        depth += 1
                    elif cj == 91:
                        depth -= 1
                        if depth == 0:
                            break
                    j -= 1
                if j < 0:
                    return steps + 1, ncopy, TERM_UNMATCHED
                ip = j
        # anything else: no-op
        ip += 1
        steps += 1
    return steps, ncopy, TERM_BUDGET


def run_region_py(buf, ip, h0, h1, k):
    """Pure-Python reference implementation of :func:`run_region`.

    Kept deliberately naive; the test-suite cross-checks the compiled kernel
    against it on random programs.
    """
    n = len(buf)
    steps = 0
    ncopy = 0
    while steps < k:
        if ip < 0 or ip >= n:
            return steps, ncopy, TERM_OFF_REGION
        c = int(buf[ip])
        if c == OP_H0_INC:
            h0 = (h0 + 1) % n
        elif c == OP_H0_DEC:
            h0 = (h0 - 1) % n
        elif c == OP_H1_INC:
            h1 = (h1 + 1) % n
        elif c == OP_H1_DEC:
            h1 = (h1 - 1) % n
        elif c == OP_INC:
            buf[h0] = (int(buf[h0]) + 1) % 256
        elif c == OP_DEC:
            buf[h0] = (int(buf[h0]) - 1) % 256
        elif c == OP_COPY_01:
            buf[h1] = buf[h0]
            ncopy += 1
        elif c == OP_COPY_10:
            buf[h0] = buf[h1]
            ncopy += 1
        elif c == OP_LOOP_BEG:
            if buf[h0] == 0:
                depth, j = 1, ip + 1
                while j < n:
                    if buf[j] == OP_LOOP_BEG:
                        depth += 1
                    elif buf[j] == OP_LOOP_END:
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                if j >= n:
                    return steps + 1, ncopy, TERM_UNMATCHED
                ip = j
        elif c == OP_LOOP_END:
            if buf[h0] != 0:
                depth, j = 1, ip - 1
                while j >= 0:
                    if buf[j] == OP_LOOP_END:
                        depth += 1
                    elif buf[j] == OP_LOOP_BEG:
                        depth -= 1
                        if depth == 0:
                            break
                    j -= 1
                if j < 0:
                    return steps + 1, ncopy, TERM_UNMATCHED
                ip = j
        ip += 1
        steps += 1
    return steps, ncopy, TERM_BUDGET
