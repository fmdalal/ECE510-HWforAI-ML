# =============================================================================
# Makefile : cocotb simulation for mac_correct.sv
# Simulator: Icarus Verilog
# Usage    : make
# =============================================================================

SIM            ?= icarus
TOPLEVEL_LANG  ?= verilog
VERILOG_SOURCES = $(PWD)/mac_correct.sv
TOPLEVEL        = mac
MODULE          = test_mac
COMPILE_ARGS   += -g2012

# WAVES=1 tells cocotb+Icarus to auto-generate dump.vcd
WAVES            = 1
COCOTB_RESOLVE_X = ZEROS

include $(shell cocotb-config --makefiles)/Makefile.sim

# Save simulation output to sim_log.txt after running
sim: sim_log.txt

sim_log.txt:
	$(MAKE) SIM=$(SIM) 2>&1 | tee sim_log.txt
