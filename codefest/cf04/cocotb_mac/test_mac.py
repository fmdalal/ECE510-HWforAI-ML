# =============================================================================
# File        : test_mac.py
# Description : cocotb testbench for INT8 MAC unit (mac_correct.sv)
# Test Plan   :
#   Phase 1 — a=3,  b=4  for 3 cycles  → expect 12, 24, 36
#   Phase 2 — assert rst for 1 cycle   → expect 0
#   Phase 3 — a=-5, b=2  for 2 cycles  → expect -10, -20
# =============================================================================

import ctypes
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer


def s32(val):
    """Convert cocotb port value to signed 32-bit integer."""
    return ctypes.c_int32(int(val)).value


def u8(val):
    """Convert signed Python integer to unsigned 8-bit for port driving."""
    return ctypes.c_uint8(val).value


@cocotb.test()
async def test_mac_basic(dut):

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    # -------------------------------------------------------------------------
    # Phase 0: Reset
    # Drive inputs on falling edge so they are stable before next rising edge
    # -------------------------------------------------------------------------
    dut.rst.value = 1
    dut.a.value   = 0
    dut.b.value   = 0
   
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)  # settle before checking
    assert s32(dut.out.value) == 0, \
        f"Reset failed: expected 0, got {s32(dut.out.value)}"
    # Release reset
    dut.rst.value = 0

    # -------------------------------------------------------------------------
    # Phase 1: a=3, b=4 for 3 cycles → expect 12, 24, 36
    # -------------------------------------------------------------------------
    dut.a.value = u8(3)
    dut.b.value = u8(4)
    for expected in [12, 24, 36]:
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)  # sample after falling edge — output settled
        assert s32(dut.out.value) == expected, \
            f"Phase 1 failed: expected {expected}, got {s32(dut.out.value)}"
        
    # -------------------------------------------------------------------------
    # Phase 2: Assert rst → expect 0
    # -------------------------------------------------------------------------
    dut.rst.value = 1
    dut.a.value   = 0
    dut.b.value   = 0

    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    assert s32(dut.out.value) == 0, \
        f"Mid-reset failed: expected 0, got {s32(dut.out.value)}"

    # -------------------------------------------------------------------------
    # Phase 3: a=-5, b=2 for 2 cycles → expect -10, -20
    # -------------------------------------------------------------------------
    dut.rst.value = 0
    dut.a.value = u8(-5)
    dut.b.value = u8(2)
    for expected in [-10, -20]:
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        assert s32(dut.out.value) == expected, \
            f"Phase 3 failed: expected {expected}, got {s32(dut.out.value)}"
        
    dut._log.info("test_mac_basic passed.")


# =============================================================================
# Test 2: Overflow behavior test
# Does the accumulator saturate or wrap around?
#
# Strategy:
#   - Use a=127, b=127 (max INT8 product = 16129) to accumulate quickly
#   - Run 133144 cycles to reach 2,147,479,576 (just below 2^31-1)
#   - One more cycle pushes past 2^31-1 → observe wrap or saturate
#
# Expected (no saturation logic):
#   - Design WRAPS to a large negative value (two's complement overflow)
#   - Wrapped value = -2,147,471,591
# =============================================================================
@cocotb.test()
async def test_mac_overflow(dut):

    MAX_S32  = 2**31 - 1       # 2,147,483,647
    PRODUCT  = 127 * 127       # 16,129 per cycle
    # Number of cycles to reach 2,147,479,576 (last value before overflow)
    CYCLES_BEFORE = 133144
    VAL_BEFORE    = 2147479576
    VAL_WRAPPED   = ctypes.c_int32(VAL_BEFORE + PRODUCT).value  # -2147471591

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    # -- Reset ----------------------------------------------------------------
    await FallingEdge(dut.clk)
    dut.rst.value = 1
    dut.a.value   = 0
    dut.b.value   = 0

    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    assert s32(dut.out.value) == 0, "Reset failed"

    dut.rst.value = 0
    dut.a.value   = u8(127)
    dut.b.value   = u8(127)

    dut._log.info(f"Overflow test: accumulating {CYCLES_BEFORE} cycles "
                  f"with a=127, b=127 (product={PRODUCT})")
    dut._log.info(f"2^31 - 1 = {MAX_S32}")

    # -- Accumulate up to just before overflow --------------------------------
    for _ in range(CYCLES_BEFORE):
        await RisingEdge(dut.clk)

    await FallingEdge(dut.clk)
    val_before = s32(dut.out.value)
    dut._log.info(f"After {CYCLES_BEFORE} cycles: out = {val_before} "
                  f"(expected {VAL_BEFORE})")
    assert val_before == VAL_BEFORE, \
        f"Pre-overflow value wrong: expected {VAL_BEFORE}, got {val_before}"

    # -- One more cycle: crosses 2^31-1 → expect wrap -------------------------
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    val_after = s32(dut.out.value)
    dut._log.info(f"After {CYCLES_BEFORE+1} cycles: out = {val_after}")

    if val_after == VAL_WRAPPED:
        dut._log.info(f"OVERFLOW BEHAVIOR: WRAP (two's complement)")
        dut._log.info(f"out wrapped from {VAL_BEFORE} to {val_after}")
    elif val_after == MAX_S32:
        dut._log.info(f"OVERFLOW BEHAVIOR: SATURATE (clamped to {MAX_S32})")
    else:
        dut._log.info(f"OVERFLOW BEHAVIOR: UNKNOWN — got {val_after}")

    # Document the behavior — this design wraps, not saturates
    assert val_after == VAL_WRAPPED, \
        f"Expected wrap to {VAL_WRAPPED}, got {val_after}. " \
        f"Design may saturate instead of wrap."

    dut._log.info("test_mac_overflow passed — design WRAPS on overflow.")
