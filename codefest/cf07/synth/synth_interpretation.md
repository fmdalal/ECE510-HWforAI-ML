# Openlane 2 Interpretation
I started with my analysis in the compute_core.sv file.  
I ran openlane synthesis using this command ___openlane --to Yosys.Synthesis /work/config.json___ but this was not a complete anlysis. This did not provide me with timing analysis but was able to estimate only area.   
However, this synethsis confirmed that my RTL is synthesizable and could be converted to actual manufacturable logic gates. There were warnings that were also identified in terms of multiple driver warnings, undriven signals and port mismatches were identified, that i need to work on. And lastly it verified that my module hierarchy instantiates correctly.   
I was able to also get pre-layout area estimate of 0.358mm^2 on sky130A 130nm simulation. Also, was able to identify the design is dominated by combinational logic (67,274 cells) vs sequential (591 FFs). 

Below is the analysis from doing openlane 2 simulation on just the Chip 0 not my compute_core. 

### Clock period:
**Clock period:** 10.0 ns set in config.json file in /hdl directory.  
**Worst Case Slack:** Setup timing check was +6.924ns and Hold timing check was +0.345ns. These values are from nom_tt_025C_1v80 (nominall, typical-typical, 25 degree C, 1.8V). No timing violations were found.   
All three corners pass; worst-case hold is +0.118 ns at the fast-cold corner.

### Critical Path:

Setup critical path (worst/tightest setup slack = 6.924 ns):

Source: _0060_ — sky130_fd_sc_hd__dfrtp_1 (rising edge FF, clocked by clk)  
Sink: cfg_done output port  
Path: _0060_/Q → output3/A (buf_2) → cfg_done (out)  
Data arrival time: 0.826 ns  
Required time: 7.750 ns  
Slack: +6.924 ns  

Hold critical path (tightest hold slack = 0.345 ns):

Source: _0068_ — sky130_fd_sc_hd__dfrtp_1  
Sink: _0060_ — sky130_fd_sc_hd__dfrtp_1  
Path: _0068_/Q → _0037_/X (and4bb_1) → _0060_/D  
Data arrival time: 0.839 ns  
Required time: 0.494 ns  
Slack: +0.345 ns (tightest hold — closest to violation)  

The setup critical path (cfg_start → counter FF) goes through: buf_1 → inv_2 → and2_1 → a32o_1/mux2_1/a31o_1 → FF, taking ~2.88 ns total data path delay.

### Total Cell Area and Top 3 contributors
Total cell area: 11,081.88 µm² synthesis estimate  from stat.log 
- sky130_fd_sc_hd__buf_2:   
    Count: 2,118   
    % of Instances: 98.2%  
- sky130_fd_sc_hd__dfrtp_2   
    Count: 9  
    % of Instances: 0.42%   
- sky130_fd_sc_hd__inv_2  
    Count: 4  
    % of Instances: 0.19%

    The 2,118 buffers dominate purely by count and area — they're driven by the 3,158 unannotated (unrouted) input ports (rx_bump_data[0:511], sram_rdata[0:511], etc.) which have no parasitics annotated because they're tied to constants or left unconnected.  

    The actual post-PnR placed area is 177,511 µm² (metrics.csv design__instance__area), inside a 9,000,000 µm² die.

### Failed Constraints, Hold Violations, and Warnings Worth Investigating
**No failed constraints** — zero setup violations, zero hold violations across all paths checked.

**Key warnings worth investigating:**
- 3,158 unannotated drivers (most critical): These are all the UCIe bump inputs and SRAM data inputs — they have no SPEF parasitic annotation because they're either tied to constants or not connected to real drivers. This means power analysis and timing for paths through these signals is inaccurate.   
- 2,118 unconstrained endpoints: All the SRAM address outputs and UCIe TX data outputs have no output timing constraints. This means setup/hold for these paths was not checked. 
- Power breakdown:Clock power dominates at 62.5% — this is because the clock tree drives many buffer cells to reach the wide IO ports.     
Sequential:    38.98 µW  (33.7%)  
Combinational:  4.38 µW   (3.8%)  
Clock:         72.26 µW  (62.5%)  
Total:        115.61 µW  
