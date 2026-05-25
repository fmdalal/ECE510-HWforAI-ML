## Synthesis Notes

- I started with doing synthesis on OpenLane2. This gave me multiple errors but ultimately I ran out of memory.
- My design overall has 9 chiplets. So, instead of running OpenLane2 synthesis on the entire design I started doing sysnthesis on my design for Chiplet 0. My idea was to perform synthesis on each of the different modules and manually calculate the timing area and power. I completed the synthesis for Chiplet 0 module, however, on speaking to the professor and friends I was recommended to use the lab servers and use Cadence to perform synthesis instead.
- Cadence had a different learning curve. For OpenLane2 I had to convert my SystemVerilog files to verilog files and then continue the synthesis. For Cadence I could use my original SystemVerilog files but the script for synthesis was .tcl.
- I performed multiple runs on Cadence, however, the first 3-4 times within 2 minutes the synthesis errored out. In the synthesis.tcl script I had to instruct the synthesis to ignore those errors as warnings and contine with the synthesis. Below are the details of the messages.
```
suppress_message LBR-43       ;# demo library missing area attributes
suppress_message VLOGPT-431   ;# clear signal in sensitivity list (by design)
suppress_message CDFG-508     ;# unused flip-flop removal (expected) 
```
- The next time I ran there was an RTL change that I had to perform as Cadence gave an error saying the code was unsynthesizable. Genus cannot synthesize a mixed async/sync reset condition inside a generate block because it cannot statically resolve the sensitivity list at elaboration time. The error was one of my signals 'clear' was synchronous and 'rst_n' was asynchronous. This was resolved by creating a temporary logic variable outside the generate block with rst signal and the clear signal and that temporary variable was then used. The ternary genvar comparisons were also simplified to direct indices since M=N=K making the conditions always true.

- The next time I ran the script overnight for synthesis- it had run out of memory.
- To clear the memory error I changed the TILE_DIM from 64 to 8 and then was able to perform full synthesis.
- I tried multiple attempts to do power analysis to my Cadence synthesis. However, the library was missing in the Cadence version provided by the university. So, ultimately power synthesis was doing using Yosys.
- I also had to re-run my synthesis as the timing reports had missing setup and hold time analysis- since I had not provided instructions in my .tcl file. 

## Scope Changes:
- Converted the taylor function to softmax which was part of the original algorithm from the M2 phase of the project. I was intially trying to do a maximum 5 order taylor equation, just so as not to perform the exponent calculation required in the softmax algorithm. However, simulated data was affected negatively, hence, changed the rtl code from chiplet_9_taylor.sv to chiplet_9_softmax.sv

- I had originally created an interface for wishbone to axi. But I am now assuming I have access to AXI ports and will not need to do the conversion.