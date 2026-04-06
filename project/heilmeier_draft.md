1) What are you trying to do? Articulate your objectives using absolutely no jargon.	
   A chiplet that can perform speech-to-text transcription fast (without using the internet and needing the cloud) and without draining the battery.
   
2) How is it done today, and what are the limits of current practice?
   Speech recognition systems run on three things today-general purpose CPU, GPU or cloud server. 
   A speech recognition system, requires multiple MACs per second of audio. For a CPU-based system the MAC operation is performed serially making it slow for real-time use and it also consumes too much power. 
   A GPU-based system is faster than CPU but consumes power, thus, not making it suitable for battery powered devices such as hearing aids and wearables.
   Sending to the cloud requires reliable network connection, introducing delay.
   There are general-purpose AI chips built into smartphones but they are not optimized for speech recognition and are integrated into SoCs, thus, not available as standalone chiplets.  
   
3) What is new in your approach and why do you think it will be successful?
   The approach that I plan to do is build a dedicated chip for conformer-based Automatic Speech Recognition (ASR) model.
   This model spends a majority of its execution in performing matrix multiplication and the dedicated chiplets will contain multiplier units to perform the calculation efficiently using parallelization. This would improve speed and lower power consumption as compared to CPUs.
   Alongside, the hardware the recognition model is also proposed to be improved in two ways. Instead of using the standard conformer model, where the global speech context and local sound patterns are extracted sequentially, the idea is to extract both simultaneously through parallel pathways. This would reduce recognition errors compared to the standard approach. Second, the model weights are converted from 32-bit floating point numbers to compact 8-bit integers. This would reduce memory requirement and power consumption.
   This approach will succeed as creating dedicated chiplet for speech AI will allow faster and less power to be consumed. With an additional improvement at the algorithm level accuracy will also be improved.