# =============================================================================
# run_questa.do  —  Questa simulation script (batch mode, VCD waveform)
# =============================================================================
# Usage: vsim -c -do run_questa.do
#
# Produces:
#   sim_run.log   — full annotated transcript
#   waveform.vcd  — signal dump (open in GTKWave on local machine)
#
# To view waveform:
#   Copy waveform.vcd to local machine
#   gtkwave waveform.vcd
# =============================================================================

transcript file sim_run.log
transcript on

puts "============================================================"
puts " MHSA 9-Chiplet Accelerator - Functional Simulation"
puts " QuestaSim 2021.3_1  (batch mode)"
puts "============================================================"

# ── Work library ──────────────────────────────────────────────────────────────
if {[file exists work]} { vdel -lib work -all }
vlib work
vmap work work

# ── Compile ───────────────────────────────────────────────────────────────────
puts "Step 1: Compiling RTL..."
vlog -sv -work work +incdir+. \
    bf16_exp.sv              \
    fp32_arith.sv            \
    ucie_link.sv             \
    interface.sv             \
    chiplet_0_qkv_outproj.sv \
    chiplet_head.sv          \
    chiplet_9_softmax.sv     \
    compute_core.sv          \
    top.sv                   \
    tb_top.sv
puts "Step 1: Compilation complete."

# ── Optimise ──────────────────────────────────────────────────────────────────
puts "Step 2: Optimising (TILE_DIM=32)..."
vopt work.tb_top -o tb_top_opt \
    -G TILE_DIM=32 \
    -G D_HEAD=32   \
    -G D_MODEL=512 \
    -G NUM_HEADS=8 \
    +acc=npr
puts "Step 2: Optimisation complete."

# ── Load and run ──────────────────────────────────────────────────────────────
puts "Step 3: Loading simulation..."
vsim -c tb_top_opt -suppress 2244

puts "Step 4: Setting up VCD waveform capture..."
# VCD captures all signals — works in batch mode without GUI
vcd file waveform.vcd

# Capture key signals organised by hierarchy
# Clocks and reset
vcd add /tb_top/clk_axi
vcd add /tb_top/clk_core
vcd add /tb_top/clk_link
vcd add /tb_top/rst_n

# AXI4-Lite control signals
vcd add /tb_top/s_axil_awaddr
vcd add /tb_top/s_axil_awvalid
vcd add /tb_top/s_axil_awready
vcd add /tb_top/s_axil_wdata
vcd add /tb_top/s_axil_wvalid
vcd add /tb_top/s_axil_wready
vcd add /tb_top/s_axil_bvalid
vcd add /tb_top/s_axil_bready
vcd add /tb_top/s_axil_araddr
vcd add /tb_top/s_axil_arvalid
vcd add /tb_top/s_axil_arready
vcd add /tb_top/s_axil_rdata
vcd add /tb_top/s_axil_rvalid
vcd add /tb_top/s_axil_rready

# AXI4-Stream input (Q tile)
vcd add /tb_top/s_axis_tdata
vcd add /tb_top/s_axis_tvalid
vcd add /tb_top/s_axis_tready
vcd add /tb_top/s_axis_tlast

# AXI4-Stream output (result tile)
vcd add /tb_top/m_axis_tdata
vcd add /tb_top/m_axis_tvalid
vcd add /tb_top/m_axis_tready
vcd add /tb_top/m_axis_tlast

# Compute core control
vcd add /tb_top/u_dut/u_compute_core/cfg_start_core
vcd add /tb_top/u_dut/u_compute_core/cfg_mode_core
vcd add /tb_top/u_dut/u_compute_core/busy_r
vcd add /tb_top/u_dut/u_compute_core/sts_done_core
vcd add /tb_top/u_dut/u_compute_core/c0_done

# Chiplet 0 FSM
vcd add /tb_top/u_dut/u_compute_core/u_c0/wstate
vcd add /tb_top/u_dut/u_compute_core/u_c0/cstate
vcd add /tb_top/u_dut/u_compute_core/u_c0/weights_ready
vcd add /tb_top/u_dut/u_compute_core/u_c0/sa_data_in
vcd add /tb_top/u_dut/u_compute_core/u_c0/q_valid

# UCIe Q broadcast
vcd add /tb_top/u_dut/u_compute_core/c0_txq_bv
vcd add /tb_top/u_dut/u_compute_core/c0_txq_bump

# Head 0 FSM
vcd add /tb_top/u_dut/u_compute_core/head_gen[0]/u_head/state
vcd add /tb_top/u_dut/u_compute_core/head_gen[0]/u_head/sa_valid
vcd add /tb_top/u_dut/u_compute_core/head_gen[0]/u_head/tx_valid_i

# Softmax chiplet 9
vcd add /tb_top/u_dut/u_compute_core/u_taylor/cfg_done
vcd add /tb_top/u_dut/u_compute_core/u_taylor/pipe_done_vec
vcd add /tb_top/u_dut/u_compute_core/u_taylor/pipe_gen[0]/u_pipe/rx_valid
vcd add /tb_top/u_dut/u_compute_core/u_taylor/pipe_gen[0]/u_pipe/nr_rdy
vcd add /tb_top/u_dut/u_compute_core/u_taylor/pipe_gen[0]/u_pipe/tx_valid_i

# Context arbiter
vcd add /tb_top/u_dut/u_compute_core/arb_head
vcd add /tb_top/u_dut/u_compute_core/arb_tile_vld

# Testbench status
vcd add /tb_top/stage1_done
vcd add /tb_top/stage45_done
vcd add /tb_top/n_errors
vcd add /tb_top/hw_cycles

puts "Step 4: VCD capture configured (44 signals)."

# ── Run simulation ────────────────────────────────────────────────────────────
puts "Step 5: Running simulation..."
vcd on
run -all
vcd off
vcd flush

# ── Summary ───────────────────────────────────────────────────────────────────
puts ""
puts "============================================================"
puts " SIMULATION COMPLETE"
puts " Log file  : sim_run.log"
puts " Waveform  : waveform.vcd"
puts "   View:    gtkwave waveform.vcd"
puts "============================================================"

# Read log and print PASS/FAIL to console
set fp [open sim_run.log r]
set log [read $fp]
close $fp
if {[string match "*\nPASS*" $log] || [string match "PASS*" $log]} {
    puts "RESULT: PASS"
} elseif {[string match "*FAIL*" $log]} {
    puts "RESULT: FAIL - check sim_run.log"
} else {
    puts "RESULT: check sim_run.log for PASS/FAIL line"
}

transcript off
quit -f
