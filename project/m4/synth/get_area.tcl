# =============================================================
#  get_area.tcl  —  Physical area report from existing netlist
#  Legacy UI mode (genus -legacy_ui)
#  Run: genus -legacy_ui -batch -f get_area.tcl
#  Runtime: ~2-5 minutes (no re-synthesis)
# =============================================================

# ── 1. Library and LEF setup ─────────────────────────────────
set_attribute lib_search_path \
    /pkgs/cadence-09-2015/SSV151/share/FoundationFlows/EXAMPLES/TEMPUS/GPDK/LIBS/GPDK045/timing/

set_attribute library typical.lib

set_attribute lef_library \
    /pkgs/cadence-09-2015/SSV151/share/FoundationFlows/EXAMPLES/TEMPUS/GPDK/LIBS/GPDK045/gsclib045.lef

# ── 2. Read the existing synthesised netlist ─────────────────
read_hdl -netlist /u/fchikani/m4/genus_output/top_netlist.v

# ── 3. Elaborate ─────────────────────────────────────────────
elaborate

# ── 4. Generate physical area report ─────────────────────────
report area \
    > /u/fchikani/m4/genus_output/area_report_physical.rpt

puts "============================================"
puts " area_report_physical.rpt written"
puts "============================================"
exit
