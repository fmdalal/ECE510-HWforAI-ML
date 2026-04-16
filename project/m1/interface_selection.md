# Interface Analysis

### Calculating Interface BandWidth 
* Target throughput for SelfAttention = 4000 GFLOPs (as shown in the hypothetical accelerator on the roofline graph).  
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