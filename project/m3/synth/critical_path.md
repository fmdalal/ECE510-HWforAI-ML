# Critical Path

```text
compute_core.sv
 └── u_host_tx (ucie_tx instance)
       │
       └── ucie_link.sv
           ├── flit_cnt[1]  ← startpoint register
           │   logic that decides which word goes into which
           │   bit position of the 512-bit flit
           └── bump_data[13] ← endpoint register

The critical path is entirely inside the ucie_tx module in ucie_link.sv — specifically the FSM and datapath logic that packs BF16 words into 512-bit flits and drives the bump pads.
```

The critical path in the synthesized compute_core design runs from the flip-flop u_host_tx_flit_cnt_reg[1] (startpoint) to the scan flip-flop u_host_tx_flit_cnt_reg[13] (endpoint), both clocked on the rising edge of clk_core (1 GHz, 1000 ps period). The path represents the flit counter logic inside the UCIe transmitter (ucie_tx) — specifically the combinational logic that computes the next value of the bump pad shift register based on the current flit count. The total data path delay is 919 ps against a required time of 919 ps, leaving a slack of exactly 0 ps — making it the critical path because it has the least timing margin of all paths in the design. The path traverses 34 logic stages including a DFFRHQX4 launch flip-flop, followed by a chain of NOR2, CLKAND2, CLKINVX, AOI21, NAND2, NOR2, MXI2, and OAI21 gates before arriving at the SDFFRHQX8 capture flip-flop. The large number of mux stages (MXI2X1) in the middle of the path indicates that the flit packing logic — which selects which 16-bit BF16 word to place into each bit position of the 512-bit bump data register — is the bottleneck.  

What would shorten the critical path?  
 This path could be shortened by:  
 (1) pipelining — inserting a register stage in the middle of the flit packing logic to split the 34-stage path into two ~17-stage paths.  
 (2) retiming — moving registers across combinational logic boundaries to balance path depths.  
 (3) upsizing critical cells such as the MXI2X1 muxes to larger drive strength variants (MXI2X2 or MXI2X4) to reduce transition times, which syn_opt would do automatically if added to the synthesis flow.

