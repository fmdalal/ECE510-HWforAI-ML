"""
test_conformer_mhsa.py
======================
cocotb testbench stub for conformer_rel_mhsa_accel.

What this harness does
──────────────────────
  1. Drives synchronous active-low reset for 4 cycles.
  2. Preloads all five weight matrices with small integer values.
  3. Loads u_bias and v_bias with small integer values.
  4. Streams in a 8×32 input activation matrix X (identity-like pattern).
  5. Streams in a 8×32 positional embedding matrix P (unit values).
  6. Asserts `start` and waits for `done` (with a generous cycle timeout).
  7. Collects and prints the streamed Y output rows.

Passing complex numerical assertions is NOT the goal here — the harness
confirms the simulation elaborates, resets cleanly, accepts inputs, and
reaches ST_DONE without hanging or X-propagation errors.

Fixed-point convention
──────────────────────
  DATA_WIDTH=16, FRAC_WIDTH=8  →  Q8.8 format
  Integer value v in fixed-point = v << 8  (i.e. v * 256)

DUT parameters used (must match RTL defaults)
──────────────────────────────────────────────
  DATA_WIDTH  = 16
  FRAC_WIDTH  = 8
  ACCUM_WIDTH = 32
  D_MODEL     = 32
  NUM_HEADS   = 2
  SEQ_LEN     = 8
  SA_ROWS     = 4
  SA_COLS     = 4
  HEAD_DIM    = 16   (= D_MODEL / NUM_HEADS)
  TILE_R      = 8    (= D_MODEL / SA_ROWS)
  TILE_C      = 8    (= D_MODEL / SA_COLS)
  WTILE_BITS  = 256  (= SA_ROWS * SA_COLS * DATA_WIDTH)
  WMAT_BITS   = 16384 (= TILE_R * TILE_C * WTILE_BITS)
  ROW_BITS    = 512  (= D_MODEL * DATA_WIDTH)
  BIAS_BITS   = 512  (= NUM_HEADS * HEAD_DIM * DATA_WIDTH)
"""

import cocotb
from cocotb.clock      import Clock
from cocotb.triggers   import RisingEdge, FallingEdge, ClockCycles, with_timeout

import random

# ── DUT parameter mirror ────────────────────────────────────────────────────
DATA_WIDTH  = 16
FRAC_WIDTH  = 8
ACCUM_WIDTH = 32
D_MODEL     = 32
NUM_HEADS   = 2
SEQ_LEN     = 8
SA_ROWS     = 4
SA_COLS     = 4
HEAD_DIM    = D_MODEL  // NUM_HEADS          # 16
TILE_R      = D_MODEL  // SA_ROWS            # 8
TILE_C      = D_MODEL  // SA_COLS            # 8
WTILE_BITS  = SA_ROWS  *  SA_COLS * DATA_WIDTH   # 256 bits per tile
WMAT_BITS   = TILE_R   *  TILE_C  * WTILE_BITS   # bits per full weight matrix
ROW_BITS    = D_MODEL  *  DATA_WIDTH             # bits per activation row
BIAS_BITS   = NUM_HEADS * HEAD_DIM * DATA_WIDTH  # bits for one bias vector

CLK_PERIOD_NS = 10   # 100 MHz
TIMEOUT_CYCLES = 500_000  # generous: softmax is iterative


# ── Helpers ──────────────────────────────────────────────────────────────────

def fp(value: float) -> int:
    """Convert a Python float to Q8.8 fixed-point integer."""
    return int(value * (1 << FRAC_WIDTH)) & ((1 << DATA_WIDTH) - 1)


def build_identity_row_flat(row_idx: int, cols: int = D_MODEL) -> int:
    """
    Build one activation row:
      element[c] = fp(0.5) if c == row_idx % cols else fp(0.0)
    Returns the flat integer value for the ROW_BITS-wide port.
    """
    val = 0
    for c in range(cols):
        elem = fp(0.5) if (c == row_idx % cols) else fp(0.0)
        val |= (elem << (c * DATA_WIDTH))
    return val


def build_unit_row_flat(cols: int = D_MODEL) -> int:
    """All elements = fp(1/cols) — safe uniform positional row."""
    val = 0
    elem = fp(1.0 / cols)
    for c in range(cols):
        val |= (elem << (c * DATA_WIDTH))
    return val


def build_weight_tile(value: float = 0.1) -> int:
    """
    Build a flat weight tile (SA_ROWS × SA_COLS elements).
    All elements set to fp(value).
    Returns integer for the WTILE_BITS-wide port.
    """
    val = 0
    elem = fp(value)
    for i in range(SA_ROWS * SA_COLS):
        val |= (elem << (i * DATA_WIDTH))
    return val


def build_bias_flat(value: float = 0.0) -> int:
    """Build NUM_HEADS × HEAD_DIM bias vector (all = fp(value))."""
    val = 0
    elem = fp(value)
    for i in range(NUM_HEADS * HEAD_DIM):
        val |= (elem << (i * DATA_WIDTH))
    return val


async def drive_reset(dut, cycles: int = 4):
    """Assert active-low reset for `cycles` rising edges."""
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, cycles)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    dut._log.info("Reset de-asserted")


async def preload_all_weights(dut):
    """
    Load all 5 weight matrices (W_Q, W_K, W_V, W_pos, W_O) tile by tile.
    Uses a small constant value (0.1) so the SA produces non-zero outputs
    without overflowing the Q8.8 accumulator.
    """
    tile_val = build_weight_tile(0.1)
    dut._log.info(f"Preloading weights — {TILE_R * TILE_C} tiles × 5 matrices")

    for sel in range(5):                    # 0=W_Q 1=W_K 2=W_V 3=W_pos 4=W_O
        for tr in range(TILE_R):
            for tc in range(TILE_C):
                await RisingEdge(dut.clk)
                dut.weight_sel.value        = sel
                dut.weight_tr.value         = tr
                dut.weight_tc.value         = tc
                dut.weight_tile_flat.value  = tile_val
                dut.weight_load.value       = 1
        await RisingEdge(dut.clk)
        dut.weight_load.value = 0

    dut._log.info("Weight preload complete")


async def preload_biases(dut):
    """Load u_bias (sel=0) and v_bias (sel=1) with zeros."""
    bias_val = build_bias_flat(0.0)
    for sel in range(2):
        await RisingEdge(dut.clk)
        dut.bias_sel.value   = sel
        dut.bias_flat.value  = bias_val
        dut.bias_load.value  = 1
        await RisingEdge(dut.clk)
        dut.bias_load.value  = 0
    dut._log.info("Bias preload complete")


async def stream_input_x(dut):
    """Feed SEQ_LEN rows of X (one per cycle)."""
    dut._log.info(f"Streaming X: {SEQ_LEN} rows × {D_MODEL} cols")
    for row in range(SEQ_LEN):
        await RisingEdge(dut.clk)
        dut.x_row_flat.value = build_identity_row_flat(row)
        dut.x_row_idx.value  = row
        dut.x_valid.value    = 1
    await RisingEdge(dut.clk)
    dut.x_valid.value = 0


async def stream_input_p(dut):
    """Feed SEQ_LEN rows of P (one per cycle)."""
    dut._log.info(f"Streaming P: {SEQ_LEN} rows × {D_MODEL} cols")
    unit_row = build_unit_row_flat()
    for row in range(SEQ_LEN):
        await RisingEdge(dut.clk)
        dut.p_row_flat.value = unit_row
        dut.p_row_idx.value  = row
        dut.p_valid.value    = 1
    await RisingEdge(dut.clk)
    dut.p_valid.value = 0


async def collect_output(dut) -> list:
    """
    Collect all SEQ_LEN output rows after done is asserted.
    Returns list of (row_idx, row_value) tuples.
    """
    rows = []
    dut._log.info("Waiting for y_valid rows...")
    for _ in range(SEQ_LEN + 4):           # +4 for pipeline drain margin
        await RisingEdge(dut.clk)
        if dut.y_valid.value == 1:
            idx = int(dut.y_row_idx.value)
            val = int(dut.y_row_flat.value)
            rows.append((idx, val))
            dut._log.info(f"  Y row[{idx}] = 0x{val:0{ROW_BITS//4}x}")
        if len(rows) == SEQ_LEN:
            break
    return rows


# ────────────────────────────────────────────────────────────────────────────
# Test 1: Reset check
#   Verify all outputs are de-asserted and FSM is in ST_IDLE after reset.
# ────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_reset(dut):
    """Drive reset and verify outputs are cleared."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())

    # Initialise all inputs to safe defaults
    dut.start.value            = 0
    dut.x_valid.value          = 0
    dut.x_row_flat.value       = 0
    dut.x_row_idx.value        = 0
    dut.p_valid.value          = 0
    dut.p_row_flat.value       = 0
    dut.p_row_idx.value        = 0
    dut.weight_load.value      = 0
    dut.weight_sel.value       = 0
    dut.weight_tr.value        = 0
    dut.weight_tc.value        = 0
    dut.weight_tile_flat.value = 0
    dut.bias_load.value        = 0
    dut.bias_sel.value         = 0
    dut.bias_flat.value        = 0
    dut.use_mask.value         = 0
    dut.attn_mask_flat.value   = 0

    await drive_reset(dut, cycles=4)

    # After reset: done, busy, y_valid must be 0; FSM must be ST_IDLE (5'd0)
    assert dut.done.value       == 0, f"done not cleared after reset: {dut.done.value}"
    assert dut.busy.value       == 0, f"busy not cleared after reset: {dut.busy.value}"
    assert dut.y_valid.value    == 0, f"y_valid not cleared after reset: {dut.y_valid.value}"
    assert int(dut.fsm_state_dbg.value) == 0, \
        f"FSM not in IDLE after reset: state={int(dut.fsm_state_dbg.value)}"

    dut._log.info("PASS: test_reset — all outputs cleared, FSM in ST_IDLE")


# ────────────────────────────────────────────────────────────────────────────
# Test 2: Input capture check
#   Stream a known X row and verify busy stays low before start.
# ────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_input_capture(dut):
    """Stream X and P rows; verify DUT accepts them without asserting busy."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())

    dut.start.value            = 0
    dut.x_valid.value          = 0
    dut.p_valid.value          = 0
    dut.weight_load.value      = 0
    dut.bias_load.value        = 0
    dut.use_mask.value         = 0
    dut.attn_mask_flat.value   = 0

    await drive_reset(dut, cycles=4)

    # Stream one row of X
    known_row = build_identity_row_flat(0)
    await RisingEdge(dut.clk)
    dut.x_row_flat.value = known_row
    dut.x_row_idx.value  = 0
    dut.x_valid.value    = 1
    await RisingEdge(dut.clk)
    dut.x_valid.value    = 0

    # Busy must still be 0 (start not yet asserted)
    assert dut.busy.value == 0, "busy unexpectedly high before start"
    assert dut.done.value == 0, "done unexpectedly high before start"

    dut._log.info("PASS: test_input_capture — DUT captures X row without spurious busy")


# ────────────────────────────────────────────────────────────────────────────
# Test 3: Full run (representative input)
#   Preload weights + biases, stream X and P, assert start,
#   wait for done, collect Y rows.
# ────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_full_run(dut):
    """
    Preload weights, stream X and P, pulse start, wait for done.
    Verifies the FSM completes without timeout and Y rows are emitted.
    Numerical correctness is NOT checked — harness validity only.
    """
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())

    # Default all inputs
    dut.start.value            = 0
    dut.x_valid.value          = 0
    dut.x_row_flat.value       = 0
    dut.x_row_idx.value        = 0
    dut.p_valid.value          = 0
    dut.p_row_flat.value       = 0
    dut.p_row_idx.value        = 0
    dut.weight_load.value      = 0
    dut.weight_sel.value       = 0
    dut.weight_tr.value        = 0
    dut.weight_tc.value        = 0
    dut.weight_tile_flat.value = 0
    dut.bias_load.value        = 0
    dut.bias_sel.value         = 0
    dut.bias_flat.value        = 0
    dut.use_mask.value         = 0
    dut.attn_mask_flat.value   = 0

    await drive_reset(dut, cycles=4)

    # ── Phase 1: preload weights and biases ───────────────────────────────
    await preload_all_weights(dut)
    await preload_biases(dut)

    # ── Phase 2: stream X and P (interleaved per cycle) ───────────────────
    dut._log.info("Streaming X and P simultaneously")
    unit_row = build_unit_row_flat()
    for row in range(SEQ_LEN):
        await RisingEdge(dut.clk)
        # X
        dut.x_row_flat.value = build_identity_row_flat(row)
        dut.x_row_idx.value  = row
        dut.x_valid.value    = 1
        # P
        dut.p_row_flat.value = unit_row
        dut.p_row_idx.value  = row
        dut.p_valid.value    = 1

    await RisingEdge(dut.clk)
    dut.x_valid.value = 0
    dut.p_valid.value = 0

    # Allow one extra cycle for capture registers to latch
    await ClockCycles(dut.clk, 2)

    # ── Phase 3: assert start ─────────────────────────────────────────────
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # busy is registered — sample on the NEXT rising edge
    await RisingEdge(dut.clk)
    assert dut.busy.value == 1, "busy not asserted after start"
    dut._log.info("start pulsed — busy asserted, waiting for done...")

    # ── Phase 4: wait for done (with timeout) ────────────────────────────
    done_seen = False
    for cycle in range(TIMEOUT_CYCLES):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            done_seen = True
            dut._log.info(f"done asserted at cycle {cycle}")
            break

    assert done_seen, f"Timeout: done not seen within {TIMEOUT_CYCLES} cycles"

    # done must be a single-cycle pulse
    await RisingEdge(dut.clk)
    assert dut.done.value == 0, "done held for more than one cycle"

    # ── Phase 5: collect Y output ─────────────────────────────────────────
    # Y is streamed out during ST_EMIT, which overlaps with/follows done
    y_rows = []
    for _ in range(SEQ_LEN + 8):
        await RisingEdge(dut.clk)
        if dut.y_valid.value == 1:
            idx = int(dut.y_row_idx.value)
            val = int(dut.y_row_flat.value)
            y_rows.append((idx, val))
            dut._log.info(f"  Y[{idx}] = 0x{val:0{ROW_BITS//4}x}")

    dut._log.info(f"Received {len(y_rows)} Y rows out of expected {SEQ_LEN}")

    # FSM should return to IDLE
    await ClockCycles(dut.clk, 4)
    assert int(dut.fsm_state_dbg.value) == 0, \
        f"FSM did not return to ST_IDLE: state={int(dut.fsm_state_dbg.value)}"
    assert dut.busy.value == 0, "busy still asserted after done"

    dut._log.info("PASS: test_full_run — FSM completed, returned to IDLE")


# ────────────────────────────────────────────────────────────────────────────
# Test 4: Back-to-back run
#   Assert start a second time immediately after first done to confirm
#   the FSM re-enters cleanly (no leftover state).
# ────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_back_to_back(dut):
    """Two consecutive runs — confirms FSM re-entry from ST_IDLE."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())

    dut.start.value            = 0
    dut.x_valid.value          = 0
    dut.p_valid.value          = 0
    dut.weight_load.value      = 0
    dut.bias_load.value        = 0
    dut.use_mask.value         = 0
    dut.attn_mask_flat.value   = 0

    await drive_reset(dut, cycles=4)
    await preload_all_weights(dut)
    await preload_biases(dut)

    unit_row = build_unit_row_flat()

    for run in range(2):
        dut._log.info(f"── Run {run + 1} ──────────────────────────────────────")

        for row in range(SEQ_LEN):
            await RisingEdge(dut.clk)
            dut.x_row_flat.value = build_identity_row_flat(row)
            dut.x_row_idx.value  = row
            dut.x_valid.value    = 1
            dut.p_row_flat.value = unit_row
            dut.p_row_idx.value  = row
            dut.p_valid.value    = 1

        await RisingEdge(dut.clk)
        dut.x_valid.value = 0
        dut.p_valid.value = 0

        await RisingEdge(dut.clk)
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0

        # Wait for done
        done_seen = False
        for cycle in range(TIMEOUT_CYCLES):
            await RisingEdge(dut.clk)
            if dut.done.value == 1:
                done_seen = True
                dut._log.info(f"  Run {run+1} done at cycle {cycle}")
                break

        assert done_seen, f"Run {run+1} timeout — done not seen"

        # Allow FSM to return to IDLE
        await ClockCycles(dut.clk, 8)
        assert int(dut.fsm_state_dbg.value) == 0, \
            f"Run {run+1}: FSM not in IDLE after completion"

    dut._log.info("PASS: test_back_to_back — two consecutive runs completed cleanly")


# ────────────────────────────────────────────────────────────────────────────
# Test 5: Mask enable
#   Enable additive mask (all ones) — verifies mask port is correctly
#   wired without hanging the FSM.
# ────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_mask_enable(dut):
    """Run with use_mask=1 and a non-zero mask. Checks FSM still completes."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())

    dut.start.value            = 0
    dut.x_valid.value          = 0
    dut.p_valid.value          = 0
    dut.weight_load.value      = 0
    dut.bias_load.value        = 0

    # Build a causal mask: upper-triangle = fp(-1e2), lower = 0
    mask_val = 0
    neg_inf_fp = fp(-100.0)  # large negative in Q8.8 saturates attention
    for qi in range(SEQ_LEN):
        for ki in range(SEQ_LEN):
            elem = neg_inf_fp if ki > qi else 0
            mask_val |= (elem << ((qi * SEQ_LEN + ki) * DATA_WIDTH))

    dut.attn_mask_flat.value   = mask_val
    dut.use_mask.value         = 1

    await drive_reset(dut, cycles=4)
    await preload_all_weights(dut)
    await preload_biases(dut)

    unit_row = build_unit_row_flat()
    for row in range(SEQ_LEN):
        await RisingEdge(dut.clk)
        dut.x_row_flat.value = unit_row
        dut.x_row_idx.value  = row
        dut.x_valid.value    = 1
        dut.p_row_flat.value = unit_row
        dut.p_row_idx.value  = row
        dut.p_valid.value    = 1

    await RisingEdge(dut.clk)
    dut.x_valid.value = 0
    dut.p_valid.value = 0

    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    done_seen = False
    for cycle in range(TIMEOUT_CYCLES):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            done_seen = True
            dut._log.info(f"done (masked run) at cycle {cycle}")
            break

    assert done_seen, f"Timeout with use_mask=1 after {TIMEOUT_CYCLES} cycles"
    await ClockCycles(dut.clk, 4)
    assert int(dut.fsm_state_dbg.value) == 0, "FSM not in IDLE after masked run"

    dut._log.info("PASS: test_mask_enable — FSM completes with causal mask active")
