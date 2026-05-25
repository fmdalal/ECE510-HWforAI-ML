# Folders and Files in M3 directory

## Files in m3 dir
### synthesis_notes.md
**Description:**  This file provides in-detail the number of times synthesis was attempted and what was the workaround to get the synthesis completed. Scope changes on the project are also mentioned.

## synth folder in m3 dir
**Path:**   
m3 -> synth

### synthesize.tcl
**Description:**  This file is the script that automates the entire synthesis flow in Cadence.   
 This file is equivalent tto config.json file.

 ### synthesis_run.log
**Description:** This file is the Cadence run log, detailing the warnings, information and errors generated. 

 ### timing_report.txt
**Description:** This file gives the timing report for synthesis run on Cadence. It details the slack, hold and setup time in detail.

 ### area_report.txt
**Description:** This file gives the total cell count. Area is not available as GPDK045 library provided by Cadence has no area data.

 ### critical_path.md
**Description:** This files gives details from the timing report about the starting and ending points and logic stages on the critical path.

 ### power_report.txt
**Description:** This file gives the power estimate. The power analysis had to be done in Yosys and not Cadence as no area data was available in Cadence.

## rtl folder in m3 dir
**Path:**   
m3 -> rtl

### top.sv
**Description:** This file gives the top-level model that instantiates with the interface and the compute_core.sv file.   

This folder has many more RTL files called by top.sv file for executing the hardware accelerator.

## tb folder in m3 dir
**Path:**   
m3 -> tb

### tb_top.sv
**Description:** This file gives the testbench for the behavioural model.

## sim folder in m3 dir
**Path:**   
m3 -> sim

### cosim_waveform.png
**Description:** This file gives the waveform generated in gtkwave. It clearly distinguishes the host-side write, internal compute and host-side read regions.

### cosim_run.log
**Description:** This file gives the output of all the testcases of the testbench and whether they passed or failed. If all tests pass the final decleartion is Pass.

## How to use the Simulator
- I have simulated using Mentor Questa 2021.3_1 version.
- In rtl folder there is a file run_sim.do. This file is a script file to run the simulation.
- **vsim -c -do run_sim.do** this is the command used to run the simulation on Centos 7 OS of the lab computers.

## How to use the Synthesis Run
- I have done my synthesis using Cadence-2022-09 version.
- In synth folder synthesize.tcl file is present. This is an automated script to perform synthesis.
- **genus -batch -files synthesize.tcl 2>&1 | tee synthesis_run.log** this is the command line used to start the synthesis.