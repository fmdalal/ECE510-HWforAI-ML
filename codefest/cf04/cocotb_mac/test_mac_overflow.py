# =============================================================================
# File        : test_mac_overflow.py
# Description : cocotb overflow behavior test for INT8 MAC unit
# Test Plan   :
#   - Use a=127, b=127 (max INT8 product=16129) to accumulate quickly
#   - Run 133144 cycles to reach 2,147,479,576 (just below 2^31-1)
#   - One more cycle pushes past 2^31-1 → observe wrap or saturate
# Expected    : WRAP — design has no saturation logic
# =============================================================================

import ctypes
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge


def s32(val):
    """Convert cocotb port value to signed 32-bit integer."""
    return ctypes.c_int32(int(val)).value


def u8(val):
    """Convert signed Python integer to unsigned 8-bit for port driving."""
    return ctypes.c_uint8(val).value


@cocotb.test()
async def test_mac_overflow(dut):

    MAX_S32       = 2**31 - 1        # 2,147,483,647
    PRODUCT       = 127 * 127        # 16,129 per cycle
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
    result = s32(dut.out.value)
    dut._log.info(f"[RESET     ] rst=1 | a=  0, b=  0 | out={result}")
    assert result == 0, f"Reset failed: expected 0, got {result}"

    dut.rst.value = 0
    dut.a.value   = u8(127)
    dut.b.value   = u8(127)

    dut._log.info(f"2^31 - 1   = {MAX_S32}")
    dut._log.info(f"product    = {PRODUCT} (127 x 127)")
    dut._log.info(f"Accumulating {CYCLES_BEFORE} cycles to reach {VAL_BEFORE}")

    # -- Accumulate up to just before overflow --------------------------------
    for _ in range(CYCLES_BEFORE):
        await RisingEdge(dut.clk)

    await FallingEdge(dut.clk)
    val_before = s32(dut.out.value)
    dut._log.info(f"After {CYCLES_BEFORE} cycles : out = {val_before} "
                  f"(expected {VAL_BEFORE})")
    assert val_before == VAL_BEFORE, \
        f"Pre-overflow value wrong: expected {VAL_BEFORE}, got {val_before}"

    # -- One more cycle: crosses 2^31-1 → observe wrap or saturate -----------
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    val_after = s32(dut.out.value)
    dut._log.info(f"After {CYCLES_BEFORE+1} cycles: out = {val_after}")

    if val_after == VAL_WRAPPED:
        dut._log.info("OVERFLOW BEHAVIOR: WRAP (two's complement)")
        dut._log.info(f"out wrapped from {VAL_BEFORE} → {val_after}")
    elif val_after == MAX_S32:
        dut._log.info(f"OVERFLOW BEHAVIOR: SATURATE (clamped to {MAX_S32})")
    else:
        dut._log.info(f"OVERFLOW BEHAVIOR: UNKNOWN — got {val_after}")

    assert val_after == VAL_WRAPPED, \
        f"Expected wrap to {VAL_WRAPPED}, got {val_after}."

    dut._log.info("test_mac_overflow passed — design WRAPS on overflow.")
