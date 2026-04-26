# ── cocotb / Icarus Verilog Makefile ─────────────────────────────────────────
# Usage:
#   make              – run all tests
#   make TESTCASE=test_reset   – run one test
#   make waves        – open GTKWave on the generated VCD

TOPLEVEL_LANG   = verilog
VERILOG_SOURCES = $(PWD)/conformer_rel_mhsa_accel.sv
TOPLEVEL        = conformer_rel_mhsa_accel
MODULE          = test_conformer_mhsa
SIM             = icarus

# Enable VCD waveform dump
COMPILE_ARGS    += -g2012
PLUSARGS        += +dumpfile=waves.vcd +dumpvars=0

# Reduce verbosity; uncomment for full cocotb log
# COCOTB_LOG_LEVEL = DEBUG

include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: waves clean_all
waves:
	gtkwave waves.vcd &

clean_all: clean
	rm -f waves.vcd *.xml *.log
