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
        
    dut._log.info("Waveform saved to dump.vcd")
 
 
async def _dump_vcd(dut):
    """Trigger VCD dump via Icarus $dumpfile and $dumpvars."""
    await Timer(1, unit="ns")
