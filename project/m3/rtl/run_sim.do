# ==============================================================================
# run_sim.do  --  Questa 2021.3_1 simulation script
# ==============================================================================

# Redirect all output to cosim_run.log
transcript file cosim_run.log
transcript on

if [file exists work] { vdel -all }
vlib work
vmap work work

# tb_top.sv contains both the behavioural DUT and the testbench -- one file only
vlog -work work -sv +acc -suppress 2244 tb_top.sv

vsim -voptargs=+acc -suppress 2244 work.tb_top

# ── Waveform recording ────────────────────────────────────────────────────────
# Dump all signals to a VCD file (open with GTKWave or convert to PNG)
vcd file cosim_waveform.vcd
vcd add -r /tb_top/*

log -r /*

run -all

# Close transcript
transcript off
