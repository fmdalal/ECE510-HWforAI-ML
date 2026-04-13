1. What are you trying to do? Articulate your objectives using absolutely no jargon.	

   A chiplet that can perform speech-to-text transcription fast (without using the internet and needing the cloud) and without draining the battery.
   
2. How is it done today, and what are the limits of current practice?

   Speech recognition systems run on three things today-general purpose CPU, GPU or cloud server.   
   A speech recognition system, requires a 4-block conformer encoder at a sequence length of T=1000 dominated by the SelfAttention module at 512 GFLOPs. For a CPU-based system the ridge point of 3.97 FLOPs/byte and the SelfAttention module has an arithmetic intensity of 1953 FLOPs/byte making it compute-bound. This means the bottleneck is the arithmetic throughput not memory bandwidth. The result is high latency and poor energy efficiency per inference. 
   A GPU-based system is faster than CPU but consumes power, thus, not making it suitable for battery powered devices such as hearing aids and wearables.  
   Sending to the cloud requires reliable network connection, introducing delay.  
   There are general-purpose AI chips built into smartphones but they are not optimized for speech recognition and are integrated into SoCs, thus, not available as standalone chiplets.  
   No existing platform simultaneously satisfies the three requirements of ASR - sufficient compute throughput, power and on-device operation without network dependency.
   
3. What is new in your approach and why do you think it will be successful?

   The approach that I plan to do is build a dedicated chip for the SelfAttention module in the conformer-based Automatic Speech Recognition (ASR) model.  
   This module spends a majority of its execution in performing arithmetic based linear and matrix calculations. The accelerator will be designed using systolic array each containing a multiply-accumulate unit to perform the calculations efficiently using parallelization. This would improve speed and lower power consumption as compared to CPUs.  
   Alongside, the hardware the model is also proposed to be improved in two ways. Instead of using the standard conformer model, where the global speech context and local sound patterns are extracted sequentially, the idea is to extract both simultaneously through parallel pathways. This would mean the SelfAttention module and the Convolution module running on parallel branches. This would reduce recognition errors compared to the standard approach. Second, the model weights are converted from 32-bit floating point numbers to Brain Float 16 format. This would reduce memory requirement and power consumption.  
   This approach will succeed as creating dedicated accelerator chip for speech AI will allow ~58 times speedup as seen in the roofline analysis. Switching to BF16 would halve the memory traffic and the parallel branches would reduce the sequential depth by removing one stage from the critical path.
