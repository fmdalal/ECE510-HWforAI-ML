# =============================================================================
# Makefile : cocotb simulation for mac_correct.sv
# Simulator: Icarus Verilog
# Usage    : make basic      — run basic test, save sim_log_basicTest.txt
#            make overflow   — run overflow test, save sim_log_overflowTest.txt
#            make run        — run both tests
#            make clean      — remove build artifacts
# Outputs  : sim_log_basicTest.txt, sim_log_overflowTest.txt, sim_build/mac.fst
# =============================================================================

SIM              ?= icarus
TOPLEVEL_LANG    ?= verilog
VERILOG_SOURCES   = $(PWD)/mac_correct.sv
TOPLEVEL          = mac
COMPILE_ARGS     += -g2012
COCOTB_RESOLVE_X  = ZEROS
WAVES             = 1

include $(shell cocotb-config --makefiles)/Makefile.sim

# -----------------------------------------------------------------------------
# Run basic test and save log
# -----------------------------------------------------------------------------
.PHONY: basic
basic:
	$(MAKE) sim SIM=$(SIM) MODULE=test_mac_basic COCOTB_TEST_MODULES=test_mac_basic 2>&1 | tee sim_log_basicTest.txt

# -----------------------------------------------------------------------------
# Run overflow test and save log
# -----------------------------------------------------------------------------
.PHONY: overflow
overflow:
	$(MAKE) sim SIM=$(SIM) MODULE=test_mac_overflow COCOTB_TEST_MODULES=test_mac_overflow 2>&1 | tee sim_log_overflowTest.txt

# -----------------------------------------------------------------------------
# Run both tests sequentially
# NOTE: use 'make run' not 'make all' — cocotb owns the 'all' target
# -----------------------------------------------------------------------------
.PHONY: run
run:
	$(MAKE) basic
	$(MAKE) overflow
	@echo "======================================================"
	@echo " Both tests complete."
	@echo " Logs: sim_log_basicTest.txt sim_log_overflowTest.txt"
	@echo " Waveform: sim_build/mac.fst"
	@echo "======================================================"
