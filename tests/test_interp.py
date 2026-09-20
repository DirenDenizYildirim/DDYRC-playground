"""Instruction-by-instruction tests for the byte machine.

Note a recurring trick: head0 starts where the instruction pointer starts, so
'>' walks head0 in lockstep with the code.  To get head0 onto a clean data
cell, '<' is used instead -- it wraps head0 to the far end of the region,
which is zero-filled scratch in these tests.
"""
import numpy as np
import pytest

from soup.interp import (TERM_BUDGET, TERM_OFF_REGION, TERM_UNMATCHED,
                         run_region, run_region_py)

K = 10_000


def build(program, n=16, data=None):
    buf = np.zeros(n, dtype=np.uint8)
    buf[: len(program)] = np.frombuffer(program, dtype=np.uint8)
    for i, v in (data or {}).items():
        buf[i] = v
    return buf


def run(program, n=16, ip=0, h0=0, h1=0, k=K, data=None):
    """Run under both implementations and assert they agree exactly."""
    buf = build(program, n, data)
    got = run_region(buf, ip, h0, h1, k)
    ref = build(program, n, data)
    want = run_region_py(ref, ip, h0, h1, k)
    assert got == want, "compiled kernel disagrees with the reference"
    assert np.array_equal(buf, ref)
    return buf, got[0], got[1], got[2]


# --- head movement and confinement ------------------------------------------

def test_head0_moves_right():
    buf, _, _, _ = run(b">+")
    assert buf[1] == ord("+") + 1


def test_head0_moves_left_and_wraps_to_the_end_of_the_region():
    buf, _, _, _ = run(b"<+", n=8)
    assert buf[7] == 1


def test_head1_moves_right():
    buf, _, _, _ = run(b"}.")
    assert buf[1] == ord("}")          # region[1] <- region[head0=0]


def test_head1_moves_left_and_wraps_to_the_end_of_the_region():
    buf, _, _, _ = run(b"{.", n=8)
    assert buf[7] == ord("{")


def test_head0_wraps_all_the_way_round_the_region():
    buf, _, _, _ = run(b"<<<+", n=4)   # 0 -> 3 -> 2 -> 1, then '+'
    assert buf[1] == ord("<") + 1


def test_the_two_heads_are_independent():
    # '>' then '}}' leaves head0=1, head1=2; '+' hits head0, '.' hits head1
    buf, _, ncopy, _ = run(b">}}+.")
    assert buf[1] == ord("}") + 1      # head0 cell incremented
    assert buf[2] == buf[1]            # head1 cell received it
    assert ncopy == 1


# --- arithmetic --------------------------------------------------------------

def test_increment():
    buf, _, _, _ = run(b"<+", n=8, data={7: 41})
    assert buf[7] == 42


def test_increment_wraps_mod_256():
    buf, _, _, _ = run(b"<+", n=8, data={7: 255})
    assert buf[7] == 0


def test_decrement():
    buf, _, _, _ = run(b"<-", n=8, data={7: 42})
    assert buf[7] == 41


def test_decrement_wraps_mod_256():
    buf, _, _, _ = run(b"<-", n=8, data={7: 0})
    assert buf[7] == 255


# --- copies ------------------------------------------------------------------

def test_copy_head0_to_head1():
    buf, _, ncopy, _ = run(b"<}.", n=8, data={7: 200})
    assert buf[1] == 200               # region[head1=1] <- region[head0=7]
    assert ncopy == 1


def test_copy_head1_to_head0():
    buf, _, ncopy, _ = run(b"<},", n=8, data={7: 9})
    assert buf[7] == ord("}")          # region[head0=7] <- region[head1=1]
    assert ncopy == 1


def test_copy_counts_both_directions():
    buf, _, ncopy, _ = run(b"<}.,", n=8, data={7: 200})
    assert ncopy == 2


# --- no-ops ------------------------------------------------------------------

def test_noop_bytes_cost_a_step_and_change_nothing():
    buf, steps, ncopy, term = run(b"\x00\x01\x02ABC", n=6)
    assert term == TERM_OFF_REGION
    assert steps == 6
    assert ncopy == 0
    assert list(buf) == [0, 1, 2, ord("A"), ord("B"), ord("C")]


def test_every_non_instruction_byte_is_a_noop():
    instructions = set(b"<>{}-+.,[]")
    for value in range(256):
        if value in instructions:
            continue
        buf, steps, ncopy, term = run(bytes([value]), n=3)
        assert (steps, ncopy, term) == (3, 0, TERM_OFF_REGION)
        assert list(buf) == [value, 0, 0], value


# --- brackets ----------------------------------------------------------------

def test_open_bracket_jumps_forward_when_cell_is_zero():
    buf, _, _, _ = run(b"<[+++]", n=8)       # region[head0=7] == 0
    assert buf[7] == 0                        # body skipped entirely


def test_open_bracket_falls_through_when_cell_is_nonzero():
    buf, _, _, _ = run(b"<[-]", n=8, data={7: 3})
    assert buf[7] == 0                        # decremented 3 -> 0


def test_close_bracket_loops_back_the_right_number_of_times():
    buf, steps, _, term = run(b"<[-]", n=8, data={7: 5})
    assert buf[7] == 0
    # '<', '[', then five times ('-', ']'), then four trailing no-ops
    assert steps == 1 + 1 + 5 * 2 + 4
    assert term == TERM_OFF_REGION


def test_forward_scan_skips_nested_pairs():
    # the '[' at 1 must match the ']' at 6, not the one at 4
    buf, _, _, _ = run(b"<[[+]+]", n=16)
    assert buf[15] == 0


def test_backward_scan_skips_nested_pairs():
    # the ']' at 9 must match the '[' at 2, stepping over the inner pair (4, 6)
    buf, _, _, term = run(b"<<[>[-]<-]", n=16, data={14: 2, 15: 3})
    assert (int(buf[14]), int(buf[15])) == (0, 0)
    assert term == TERM_OFF_REGION


def test_unmatched_open_bracket_terminates():
    _, _, _, term = run(b"<[+", n=8)
    assert term == TERM_UNMATCHED


def test_unmatched_close_bracket_terminates():
    _, _, _, term = run(b"<+]", n=8)
    assert term == TERM_UNMATCHED


def test_bracket_with_a_match_outside_the_region_is_unmatched():
    # the ']' lives in the region but its partner would be beyond the end
    _, _, _, term = run(b"<[+++", n=6)
    assert term == TERM_UNMATCHED


def test_matching_is_recomputed_at_execution_time():
    # the '.' writes a ']' over region[6]; the '[' at 1 then has a match
    prog = b"<[" + b"\x00" * 4 + b"\x00"
    buf = build(prog, n=8)
    buf[6] = ord("]")
    steps, _, term = run_region(buf, 0, 0, 0, K)
    assert term == TERM_OFF_REGION   # jumped to 6, then ran off the end


# --- termination -------------------------------------------------------------

def test_step_budget_exhausted():
    _, steps, _, term = run(b"<+[]", n=8, k=37)   # '[]' with a nonzero cell spins
    assert term == TERM_BUDGET
    assert steps == 37


def test_ip_runs_off_the_end_of_the_region():
    _, steps, _, term = run(b"++", n=2)
    assert term == TERM_OFF_REGION
    assert steps == 2


def test_ip_starting_outside_the_region_terminates_immediately():
    buf = np.zeros(8, dtype=np.uint8)
    steps, _, term = run_region(buf, 8, 0, 0, K)
    assert (steps, term) == (0, TERM_OFF_REGION)


def test_budget_is_checked_before_the_region_bound():
    _, steps, _, term = run(b"", n=4, k=4)
    assert (steps, term) == (4, TERM_BUDGET)


def test_ip_never_wraps_the_way_heads_do():
    # heads wrap, the instruction pointer does not: it falls off and stops
    _, steps, _, term = run(b"<<<<", n=4)
    assert (steps, term) == (4, TERM_OFF_REGION)


# --- code and data are the same memory ---------------------------------------

def test_a_program_can_overwrite_its_own_instruction():
    buf, _, _, _ = run(b"}}}.+++")
    assert buf[3] == ord("}")     # the '.' wrote over itself after executing


def test_a_write_can_create_an_instruction_that_is_then_executed():
    # '.' copies itself to region[10]; the ip later reaches 10 and copies again
    buf, _, ncopy, _ = run(b".+", n=16, h1=10)
    assert ncopy == 2
    assert buf[10] == ord(".") + 1


# --- differential test against the reference implementation ------------------

@pytest.mark.parametrize("seed", range(60))
def test_random_programs_match_the_reference_implementation(seed):
    rng = np.random.default_rng(seed)
    alphabet = np.frombuffer(b"<>{}-+.,[]" * 3 + bytes(range(60)), dtype=np.uint8)
    buf = rng.choice(alphabet, size=64).astype(np.uint8)
    ref = buf.copy()
    assert run_region(buf, 0, 0, 0, 5000) == run_region_py(ref, 0, 0, 0, 5000)
    assert np.array_equal(buf, ref)


# --- the other reading of "confined": halt instead of wrap --------------------

def test_halt_mode_ends_the_run_when_head0_leaves_the_region():
    buf = build(b"<+", n=8)
    steps, _, term = run_region(buf, 0, 0, 0, K, False)
    assert (steps, term) == (1, TERM_OFF_REGION)
    assert not buf[7]                      # the '+' never ran


def test_halt_mode_ends_the_run_when_head1_leaves_the_region():
    buf = build(b"{.", n=8)
    steps, _, term = run_region(buf, 0, 0, 0, K, False)
    assert (steps, term) == (1, TERM_OFF_REGION)


def test_halt_mode_allows_moves_that_stay_inside():
    buf = build(b">>>+", n=8)
    steps, _, term = run_region(buf, 0, 0, 0, K, False)
    assert term == TERM_OFF_REGION         # ran off the end normally
    assert buf[3] == ord("+") + 1


def test_wrap_and_halt_agree_while_no_head_leaves_the_region():
    for prog in (b">}+.", b">>}}-,", b"}}}.", b"<>{}"):
        a = build(prog, n=32)
        b = build(prog, n=32)
        # both heads start mid-region, so none of these programs reach an edge
        assert run_region(a, 0, 5, 5, K, True) == run_region(b, 0, 5, 5, K, False)
        assert np.array_equal(a, b)


def test_without_a_loop_the_head_and_the_ip_reach_the_edge_together():
    """Ring geometry: ip and head0 both start at the centre and move one cell
    per step in opposite directions, so a straight run of '<' takes them out
    of the region on the same step -- halt reports it one step earlier,
    because the head moves before the next ip bounds check."""
    n, mid = 129, 64
    buf = np.frombuffer(b"<" * n, dtype=np.uint8).copy()
    wrap = run_region(buf.copy(), mid, mid, mid, K, True)
    halt = run_region(buf.copy(), mid, mid, mid, K, False)
    assert wrap == (mid + 1, 0, TERM_OFF_REGION)
    assert halt == (mid + 1, 0, TERM_OFF_REGION)


def test_a_loop_lets_head0_reach_the_edge_and_the_two_modes_then_differ():
    # '[<]' over non-zero memory walks head0 backwards without moving the ip
    buf = np.frombuffer(b"[<]" * 40, dtype=np.uint8).copy()
    wrap = run_region(buf.copy(), 0, 60, 60, K, True)
    halt = run_region(buf.copy(), 0, 60, 60, K, False)
    assert wrap[2] == TERM_BUDGET and wrap[0] == K
    assert halt[2] == TERM_OFF_REGION and halt[0] < K


@pytest.mark.parametrize("seed", range(20))
def test_halt_mode_matches_the_reference_implementation(seed):
    rng = np.random.default_rng(seed)
    alphabet = np.frombuffer(b"<>{}-+.,[]" * 3 + bytes(range(60)), dtype=np.uint8)
    buf = rng.choice(alphabet, size=64).astype(np.uint8)
    ref = buf.copy()
    assert (run_region(buf, 0, 0, 0, 5000, False) ==
            run_region_py(ref, 0, 0, 0, 5000, False))
    assert np.array_equal(buf, ref)
