# =============================================================================
# File        : test_mac_basic.py
# Description : cocotb basic functional test for INT8 MAC unit
# Test Plan   :
#   Phase 1 — a=3,  b=4  for 3 cycles  → expect 12, 24, 36
#   Phase 2 — assert rst for 1 cycle   → expect 0
#   Phase 3 — a=-5, b=2  for 2 cycles  → expect -10, -20
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
async def test_mac_basic(dut):

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    # -- Phase 0: Reset -------------------------------------------------------
    await FallingEdge(dut.clk)
    dut.rst.value = 1
    dut.a.value   = 0
    dut.b.value   = 0

    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    result = s32(dut.out.value)
    dut._log.info(f"[RESET     ] rst=1 | a= 0, b= 0 | out={result}")
    assert result == 0, f"Reset failed: expected 0, got {result}"

    dut.rst.value = 0
    dut.a.value   = u8(3)
    dut.b.value   = u8(4)

    # -- Phase 1: a=3, b=4 for 3 cycles → 12, 24, 36 ------------------------
    dut._log.info("--- Phase 1: a=3, b=4 for 3 cycles (expect 12, 24, 36) ---")
    for expected in [12, 24, 36]:
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        result = s32(dut.out.value)
        dut._log.info(f"[CYCLE     ] rst=0 | a= 3, b= 4 | out={result}")
        assert result == expected, \
            f"Phase 1 failed: expected {expected}, got {result}"

    # -- Phase 2: Assert rst → 0 ---------------------------------------------
    dut.rst.value = 1
    dut.a.value   = 0
    dut.b.value   = 0

    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    result = s32(dut.out.value)
    dut._log.info(f"[RST ASSERT] rst=1 | a= 0, b= 0 | out={result}")
    assert result == 0, f"Mid-reset failed: expected 0, got {result}"

    # -- Phase 3: a=-5, b=2 for 2 cycles → -10, -20 -------------------------
    dut.rst.value = 0
    dut.a.value   = u8(-5)
    dut.b.value   = u8(2)

    dut._log.info("--- Phase 3: a=-5, b=2 for 2 cycles (expect -10, -20) ---")
    for expected in [-10, -20]:
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        result = s32(dut.out.value)
        dut._log.info(f"[CYCLE     ] rst=0 | a=-5, b= 2 | out={result}")
        assert result == expected, \
            f"Phase 3 failed: expected {expected}, got {result}"

    dut._log.info("test_mac_basic passed.")
