# 1. Set up target technology library
suppress_message LBR-43
suppress_message VLOGPT-431
suppress_message CDFG-508

set_db init_lib_search_path /pkgs/cadence-09-2015/SSV151/share/FoundationFlows/EXAMPLES/TEMPUS/GPDK/LIBS/GPDK045/timing/
set_db library typical.lib


# 2. Read SystemVerilog design files

read_hdl -sv {compute_core.sv chiplet_0_qkv_outproj.sv chiplet_9_softmax.sv chiplet_head.sv fp32_arith.sv interface.sv ucie_link.sv}



# 3. Elaborate the top level module

elaborate compute_core



# 4. Define constraints (10ns clock / 100MHz)

define_clock -period 1000  -name clk_core [get_ports clk_core]
define_clock -period 4000 -name clk_axi  [get_ports clk_axi]
define_clock -period 500   -name clk_link [get_ports clk_link]



# 5. Run synthesis

syn_generic

syn_map

syn_opt

# 6. Write gate-level netlist

write_netlist > gate_level_netlist.v



# 7. Export reports

redirect area_report.txt   { report_area }

redirect gates_report.txt  { report_gates }

redirect timing_report.txt { report_timing }

# 8. Power
set_switching_activity -default -static_probability 0.5 -toggle_rate 0.2
redirect power_report.txt  { report_power -hier }
redirect power_report_flat.txt {report_power}

exit
