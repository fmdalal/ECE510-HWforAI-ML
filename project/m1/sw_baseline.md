# Software Baseline

### CPU Model
Intel Core Ultra 7 Processor  
* 256V  
* 2200MHz  
* 8 Cores  
* 8 Logical Proecessors

### Operating System
Microsoft Windows 11 Home

### Memory
* Total physical memory: 15.6 GB, available: 1.10 GB
* Total virtual memory : 57.6 GB, available: 33.4 GB

### Python Version
Version: 13.13.13

### Batch Size
Batch size: 2

---------------------------------------------------------------------
### Execution Time
Measured wall-clock time from project_profile.txt
  Run  1:   30.34 ms  
  Run  2:   29.63 ms  
  Run  3:   29.72 ms  
  Run  4:   29.27 ms  
  Run  5:   30.72 ms  
  Run  6:   28.16 ms  
  Run  7:   22.32 ms  
  Run  8:   24.29 ms  
  Run  9:   22.61 ms  
  Run 10:   22.44 ms  

  Median = 28.715 ms

  ---------------------------------------------------------------------
 ### Throughput in FLOPs/sec
Meadian value = 28.715 ms for the software runtime  
T = 1000 sequence  
B = 2

Frame processed per second = (B \* T) / time  
        = (2 * 1000) / 28.715 ms
        = 69,650.00871 frames/sec

-----

### Memory Usage 
From memory profiling in project_profile.txt

Adding the CPU memory used by the individual operators and diving by 10 (total runs)  
Memory usage = 177.41 MB per run 

