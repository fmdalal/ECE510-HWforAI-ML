# =============================================================================
# synthesise.tcl  —  Cadence Genus synthesis script
# =============================================================================
# Design  : MHSA hardware accelerator (9-chiplet BF16 systolic array)
# Top     : top
# Process : update LIB_PATH and PDK variables for your target process node
# Clocks  : clk_axi 250 MHz, clk_core 1 GHz, clk_link 1 GHz
# Tool    : Cadence Genus 19.x or later
#
# Usage   : genus -f config.tcl | tee genus.log
# =============================================================================

# =============================================================================
# 0. Tool and process configuration — GPDK045 typical corner
# =============================================================================
# Redirect all Genus output to a log file
# Both screen and file output are captured
# Log file: captured via shell tee command at invocation
# Run as: genus -batch -files config.tcl 2>&1 | tee synthesis_run.log
puts "Synthesis log: synthesis_run.log (via tee)" 
puts "Start time: [clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}]"
puts "Host: [exec hostname]"
puts "============================================================"
puts " MHSA 9-Chiplet Accelerator — Genus Synthesis"
puts " GPDK045 typical.lib  |  TILE_DIM=8   |  1 GHz target"
puts "============================================================"
puts "" 
# Library search path — Cadence GPDK045 typical.lib
set_db init_lib_search_path /pkgs/cadence-09-2015/SSV151/share/FoundationFlows/EXAMPLES/TEMPUS/GPDK/LIBS/GPDK045/timing/

set DESIGN_NAME    "top"
set OUTPUT_DIR     "./genus_output"

file mkdir $OUTPUT_DIR

# =============================================================================
# 1. Technology library — MUST load before elaborate
# =============================================================================
# init_lib_search_path already set above; just name the lib file directly
set_db library typical.lib

# =============================================================================
# 2. Read RTL — bottom-up order (primitives before consumers)
# =============================================================================
# Increase loop unroll limit — default 1024 is too small for:
#   flat_in reset loop:   TILE_DIM×TILE_DIM = 32×32 = 1024 iterations (at TILE=32)
#   f_data_w reset loop:  FIFO_D×WORDS_PER_BEAT = 256×32 = 8192 iterations
set_db hdl_max_loop_limit 10000

read_hdl -sv [list \
    bf16_exp.sv          \
    fp32_arith.sv        \
    ucie_link.sv         \
    interface.sv         \
    chiplet_9_softmax.sv \
    chiplet_head.sv      \
    chiplet_0_qkv_outproj.sv \
    compute_core.sv      \
    top.sv               \
]

# =============================================================================
# 3. Elaborate — TILE_DIM=32 (64 causes OOM on server, see note below)
# =============================================================================
# Genus 19.x syntax: single -parameters flag, list of {name value} pairs
# TILE_DIM=8 for synthesis
# TILE=16 OOMs at 65GB during syn_generic partition budgeting phase.
# Memory scales as TILE^2 per array: TILE=16 used 48.7GB peak.
# TILE=8: 704 PEs × 11 arrays — estimated 15-20GB, fits on server.
# Critical path (fp32_mul → fp32_add through one PE) is IDENTICAL at all sizes.
# Gate count scaling: TILE=8 result × (64/8)^2 = × 64 = TILE=64 estimate.
# Timing (WNS) is identical — critical path depth does not change with TILE.
elaborate $DESIGN_NAME \
    -parameters {{TILE_DIM 8} {D_HEAD 8} {D_MODEL 512} {NUM_HEADS 8} {FIFO_D 256}}

check_design -unresolved

# =============================================================================
# 4. Clock definitions
# =============================================================================
# clk_axi: 250 MHz = 4 ns period, 50% duty cycle
create_clock -name clk_axi  -period 4.0  -waveform {0 2.0} \
    [get_ports clk_axi]

# clk_core: 1 GHz = 1 ns period, 50% duty cycle
create_clock -name clk_core -period 1.0  -waveform {0 0.5} \
    [get_ports clk_core]

# clk_link: 2 GHz UCIe PHY bump clock — NOW A REAL ACTIVE CLOCK
# All ucie_tx and ucie_rx bump-side FFs run on this domain.
# clk_link comes from a separate PLL (UCIe PHY PLL) — asynchronous to
# both clk_axi and clk_core.
create_clock -name clk_link -period 0.5  -waveform {0 0.25} \
    [get_ports clk_link]

# All three clocks are asynchronous to each other (separate PLLs):
#   clk_axi  (250 MHz) — host SoC PLL
#   clk_core (1 GHz)   — chiplet compute PLL
#   clk_link (2 GHz)   — UCIe PHY PLL
# Genus will not analyse timing paths that cross between any of these.
# 2-FF synchronisers in ucie_tx and ucie_rx handle all crossings.
set_clock_groups -asynchronous \
    -group [get_clocks clk_axi]  \
    -group [get_clocks clk_core] \
    -group [get_clocks clk_link]


# =============================================================================
# 5. CDC false paths
# =============================================================================
# All three clocks are asynchronous — set_clock_groups already prevents
# timing analysis across domains. Add explicit false paths as belt-and-
# suspenders for any paths the clock groups don't cover.
#
# Clock-based false paths (valid after clock creation, before synthesis):
# These cover ALL signals crossing between domains, not just named pins.
# This is more robust than pin-based paths which may not resolve post-elab.

# clk_axi <-> clk_core: all crossing paths are false
set_false_path -from [get_clocks clk_axi]  -to [get_clocks clk_core]
set_false_path -from [get_clocks clk_core] -to [get_clocks clk_axi]

# clk_core <-> clk_link: all crossing paths are false
set_false_path -from [get_clocks clk_core] -to [get_clocks clk_link]
set_false_path -from [get_clocks clk_link] -to [get_clocks clk_core]

# clk_axi <-> clk_link: all crossing paths are false
set_false_path -from [get_clocks clk_axi]  -to [get_clocks clk_link]
set_false_path -from [get_clocks clk_link] -to [get_clocks clk_axi]

# Reset: async reset crosses all domains
set_false_path -from [get_ports rst_n]

# =============================================================================
# 6. DONT_TOUCH on synchroniser flip-flops
# =============================================================================
# Clock-domain false paths (section 5) prevent Genus from optimising across
# CDC boundaries. No additional dont_touch needed — the false paths ensure
# Genus does not merge synchroniser FFs with surrounding logic.

# =============================================================================
# 7. I/O constraints
# =============================================================================
# Input delays relative to clk_axi (host interface signals)
# Note: SCALE_BF16 CSR should be written with 0x3E35 (1/sqrt(32)) at runtime
set_input_delay  -clock clk_axi  -max 1.5 [get_ports s_axil_*]
# AXI-Stream slave: tdata/tvalid/tlast/tkeep/tuser/tid are inputs, tready is output
set_input_delay  -clock clk_axi  -max 1.5 [get_ports {s_axis_tdata s_axis_tkeep s_axis_tvalid s_axis_tlast s_axis_tuser s_axis_tid}]
set_output_delay -clock clk_axi  -max 1.5 [get_ports s_axis_tready]
# AXI-Stream master: tdata/tvalid/tlast/tkeep/tuser are outputs, tready is input
set_output_delay -clock clk_axi  -max 1.5 [get_ports {m_axis_tdata m_axis_tkeep m_axis_tvalid m_axis_tlast m_axis_tuser}]
set_input_delay  -clock clk_axi  -max 1.5 [get_ports m_axis_tready]
set_output_delay -clock clk_axi  -max 1.5 [get_ports {s_axil_awready s_axil_wready
                                                           s_axil_bresp s_axil_bvalid
                                                           s_axil_arready s_axil_rdata
                                                           s_axil_rresp s_axil_rvalid}]

# Memory interface — driven/captured at clk_core rate
set_input_delay  -clock clk_core -max 0.3 [get_ports {mem_rdata mem_rvalid mem_gnt}]
set_output_delay -clock clk_core -max 0.3 [get_ports {mem_addr mem_wdata mem_wen
                                                           mem_req}]

# Reset and clocks: no timing constraints needed (async reset, clock roots)
# rst_n false path already set in section 5

# IRQ output: driven in clk_axi domain
set_output_delay -clock clk_axi -max 1.5 [get_ports irq]

# =============================================================================
# 8. Synthesis effort and optimisation
# =============================================================================
set_db syn_generic_effort   high
set_db syn_map_effort       high
set_db syn_opt_effort       high

# Allow Genus to retime FP32 adder/multiplier chains across register boundaries
# to meet the 1 GHz clk_core target
# Enable retiming — use full attribute name to avoid ambiguity
set_db / .retime_effort_level medium

# =============================================================================
# 9. Run synthesis
# =============================================================================
syn_generic
syn_map
syn_opt

# =============================================================================
# 10. Reports
# =============================================================================
report_timing -max_paths 10 -path_type full \
    > $OUTPUT_DIR/timing_report.rpt

report_power \
    > $OUTPUT_DIR/power_report.rpt

report_area \
    > $OUTPUT_DIR/area_report.rpt

report_gates \
    > $OUTPUT_DIR/gate_count_report.rpt

report_clock_gating \
    > $OUTPUT_DIR/clock_gating_report.rpt

# Check for unresolved instances and multi-driven nets
check_design -all > $OUTPUT_DIR/check_design.rpt

# =============================================================================
# 11. Write outputs
# =============================================================================
# Gate-level netlist
write_hdl > $OUTPUT_DIR/${DESIGN_NAME}_netlist.v

# Standard delay format (for post-synthesis simulation)
write_sdf -version 3.0 > $OUTPUT_DIR/${DESIGN_NAME}.sdf

# Design constraints (for place and route)
write_sdc > $OUTPUT_DIR/${DESIGN_NAME}.sdc

# Genus database (for incremental runs)
write_db $OUTPUT_DIR/${DESIGN_NAME}_genus.db

puts ""
puts "============================================================"
puts "  Synthesis complete."
puts "  End time: [clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}]"
puts "  Check $OUTPUT_DIR/ for reports and netlist."
puts "  Log file : synthesis_run.log"
puts "  Review timing_report.rpt for clk_core WNS first."
puts "  Expected: WNS >= 0 on clk_core, WNS >= 0 on clk_axi"
puts "============================================================"
