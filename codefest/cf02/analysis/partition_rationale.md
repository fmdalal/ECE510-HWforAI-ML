# HW/SW Partition Proposal

1. Which kernel(s) you will accelerate in hardware and why the roofline analysis supports that choice.

The kernel I plan to accelerate is the SelfAttention module. Observe the project_profile.txt document, specifically the 5th section in this document - Per-Module Hook Timing. This section provides the mean time with SelfAttention module taking the most mean time which across the four blocks amounts to approximately 9.9ms. This is the most compute intensive module which can also further be confirmed by observing Section 2 of the same txt document. The rows aten::bmm, aten::addmm and aten::softmax together add upto 21.0% of the CPU time taken which is slightly lower than the convolution module that takes 21.2% of the CPU time.  
The roofline analysis for the SelfAttention kernel sits on the compute bound line of my hardware, which indicates it is limited by arithmetic throughput not memory bandwidth. Its Arithmetic Intensity approximately equaling 1953FLOPs/B, with the ridge point being 3.97 FLOPs/B.  
The number of FLOPs when performing the calculation for roofline analysis shows the SelfAttention module requires close to 512 GFLOPs per forward pass for a sequence length of 1000. The number of FLOPs is driven by T<sup>2</sup> scaling in Stage 2 (QK<sup>T</sup>) and Stage 4 (Scores * V bmm) of the module. Thus, if the input sequence length increases the FLOPs needed will grow quadratically making it a dominant bottleneck.

2. What the software baseline will continue to handle?

The software baseline will continue to handle the conformer model that is not a part of the SelfAttention module. Specifically the software will handle the input subsampling before the conformer block, the Convolution module, the Feed forward module, the normalization at the end of each conformer block and the control flow and data flow movement across the modules. These modules are handled by the software as the scaling on these modules are linear. Further, the arithmetic intensity of these modules are well-matched to the CPU capability and finally keeping the modules in software preserves the flexibilty where the model parameters can be changed without redesigning the hardware.

3. What interface bandwidth your accelerator would need to avoid becoming interface-bound at the target throughput?

### Calculating Interface BandWidth 
* Target throughput for SelfAttention = 4000GFLOPs (as shown in the hypothetical accelerator on the roofline graph).  
* Bytes the accelerator needs per forward pass  - values from the ai_calculation.md document  
Total bytes for 1 block = 262,209,536 bytes  
Total bytes for 4 blocks = 1,048,838,144 bytes ~ 1GB
* Time to complete SelfAttention at target throughput  
<pre> Time = FLOPs / Peak compute  
      = 512,097,536,000 / (4000 * 10<sup>9</sup>)  
      = 0.128 seconds</pre>
* Required interface BW  
<pre> Interface BW = Bytes / time  
              = 1,048,838,144 / 0.128  
              = 8.19 GB/s </pre>  
The recommended interface that I would use is  AXI4-Lite for the control plane and AXI4-Stream 512-bit @250MHz for data plane. Below is the block diagram for the same. It uses wishbone2axi bridge as my Intel hardware does not allow a direct interface with AXI buses being a sealed SoC. Our requirement is 8.19 GB/s and the interface provides 16GB/s.
![Interface](image2.png)

4. Whether your kernel is compute-bound or memory-bound on your current hardware, and whether you expect your accelerator design to change that.

My SelfAttention module is compute-bound as can be observed in the roofline graph where its sitting on the compute-bound line of the graph. The proposed accelerator will not change the bound classification, it will continue to remain on the compute-bound section of the graph. The accelerator will change the performance ceiling. Currently the CPU celing is 544 GFLOPs with a predicted time of 0.94ms. The accelerator will modify this to be 4000 GFLOPs with a predicted time of 0.128 ms, providing a speedup of ~58 times faster.

