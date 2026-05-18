# M3 Changes 

- **Scale TILE back up — and pipeline the weight-load path**  
**Current result:** With TILE=8, synthesis produced only 9 flip-flops and 2,156 cells — of which 2,118 (98.2%) are buffers. This means at TILE=8 the actual compute logic was almost entirely optimized away by Yosys (it removed 18 unused modules including systolic_array, ucie_tx, ucie_rx). The chip area of 11,081 µm² is dominated by IO buffering, not real logic.  
**Change:** Increase to TILE=16 or TILE=32 to instantiate real systolic array logic. The full design at TILE=64 produced 67,865 cells and 358,157 µm² — scaling linearly, TILE=16 should give ~4,200 cells and ~22,000 µm², which is manageable for full PnR.

- **Change clock period at 10 ns — timing is far from binding:**  
**Current result:** Setup worst slack = +6.924 ns out of a 10 ns period. The actual critical path delay is only 3.076 ns (cfg_start → _0061_ FF through buf_1 → inv_2 → and2_1 → a32o_1). This means the design runs comfortably at 100 MHz and could be pushed to ~325 MHz without any changes.  
**Change:** Tighten the clock to 5 ns (200 MHz) to stress-test the systolic array datapath. At TILE=8 there is no real compute path to time, but at TILE=16+ the PE multiply-accumulate chain will become the critical path and this constraint will drive Yosys/ABC to optimize it. If timing closes with positive slack at 5 ns, push further to 3 ns.

- **Fix the counter width — Yosys already found the optimization**  
**Current result:** The synthesis log shows Yosys automatically found that the cnt counter only needs 6 bits (not 8):
Removed top 2 bits (of 8) from wire chiplet_0_qkv_outproj.cnt
Removed top 24 bits (of 32) from port Y of $add cell
This means the RTL declares cnt as 8-bit but only 6 bits are used (for counting up to cfg_num_tiles which is at most 64 = TILE_DIM/TILE). Yosys pruned 31 bits from the adder's B port too.  
**Change:** Fix the RTL to explicitly declare cnt as logic [5:0] matching actual usage. This removes dead code, makes intent clear, and prevents future synthesis tools from needing to rediscover this optimization.