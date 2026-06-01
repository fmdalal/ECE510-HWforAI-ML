# Remaining Tasks Before M4

## 1. Fix UCIe CRC `always_comb` loop to eliminate Icarus simulation fallback

In `ucie_link.sv` lines 13–22, the CRC-8 computation uses a `logic [7:0] c` variable
with `c[7]` and `c[6:0]` part-selects inside an `always_comb` block. Icarus Verilog 12
cannot handle part-selects on `logic` variables in `always_comb` and silently includes
all bits, producing incorrect CRC values during simulation. Replace the `logic c`
accumulator with an `integer ci` and shift via `((ci << 1) & 8'hFE)` as a workaround,
then verify CRC correctness by driving a known payload through `ucie_tx` in cocotb and
checking the 8-bit CRC field in `bump_data[15:8]` against a Python reference
implementation (`crcmod` with polynomial 0x31, init 0xFF, final XOR 0xFF).

---

## 2. Extend cocotb testbench to complete the full 5-stage MHSA pipeline and assert `done`

The current smoke test (`test_compute_core.py`) times out after 5,000 `clk_core` cycles
without observing `STATUS[1]=done`. The pipeline stall occurs because the AXI-Stream
input FIFO in `axis_input_fifo` (interface.sv) assembles `BEATS_PER_TILE=128` beats
before asserting `tile_valid`, but the UCIe TX in `ucie_tx` only holds 8 credits and
stalls after 8 flits — blocking chiplet 0 from receiving the full tile. Fix by
increasing the UCIe credit counter initial value from `4'd8` to `4'd`(FLITS) in
`ucie_tx` (ucie_link.sv line ~145), then extend `SIM_TIMEOUT_CYCLES` to 500,000 and
confirm `done=1` is asserted and the output tile is non-zero on `m_axis_tdata`.

---

## 3. Replace the systolic array's combined `rst_n_clear` signal with a synchronous clear to eliminate hold-time violations on `clk_core`

In `fp32_arith.sv` (systolic_array module, line ~257), the signal
`wire rst_n_clear = rst_n & ~clear` is used as an asynchronous reset input to
`always_ff @(posedge clk_core or negedge rst_n_clear)` inside the `acc_fb_ff` and
`drain_ff` blocks. Gating an asynchronous reset with a synchronous control signal
creates a combinational glitch path that can cause spurious resets and hold-time
violations at 1 GHz. Replace with a synchronous clear: remove `rst_n_clear`, keep
`rst_n` as the sole asynchronous reset, and add `else if (clear) acc_fp32[gi][gj] <= 32'h0`
as the first `else if` branch inside each `always_ff` block. Re-run Genus synthesis
and confirm the `clk_core` slack remains ≥ 0 ps with 0 timing violations.
