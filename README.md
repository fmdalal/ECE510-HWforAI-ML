# ECE510-HWforAI-ML
**Name:**                       Fatema Chikani

**Course:**                     ECE510 Spring 2026

**Tentative project topic:**    Conformer-baser Automatic Speech Recognition

## 1. Module to accelerate: Relative Multi Head Self Attention module
The attention module basically transforms the semantic meaning of audio sequences into a representation that is aligned contextually.

It uses Query (Q), Key (K) and Value (V) matrices to calculate the attention scores and weights providing a new context of information.

A multi head self attention is where the dimensions of Q, K and V matrices are of lower dimension by a factor of the number of multiple heads it needs.

The accelerator targets the dominant arithmetic bottleneck of the
Conformer encoder — its linear projections and matrix multiplications —
and maps them onto a **weight-stationary systolic array** of
multiply-accumulate (MAC) units. Compared to a CPU baseline the
roofline analysis predicts approximately **58× speedup** alongside
lower power consumption, since every MAC unit is active every cycle
rather than sharing a general-purpose execution pipeline.

## 2. BFloat16 (BF16) weight format

Model weights are stored and operated on in **BFloat16** rather than
the default FP32.

```
FP32:   1 sign | 8 exponent | 23 mantissa  = 32 bits
BF16:   1 sign | 8 exponent |  7 mantissa  = 16 bits
                ↑ same range as FP32
```

BF16 keeps the same 8-bit exponent as FP32, preserving the full dynamic
range of the trained weights. Only the mantissa is truncated from 23 to
7 bits. For inference on a pre-trained model this is sufficient: the
weight magnitudes are already in a well-bounded range post-training, and
the 7-bit mantissa gives ~0.8% relative error — acceptable for ASR word
error rates.

Compared to FP32, BF16 delivers:

| Metric | Improvement |
|---|---|
| Weight memory footprint | **−50%** (16 vs 32 bits per element) |
| Memory bandwidth per inference | **−50%** (fewer bytes to load per tile) |
| SRAM area for weight store | **−50%** |
| MAC datapath width | **−50%** (16-bit multipliers vs 32-bit) |
| DSP slice count | **~−50%** on Xilinx Ultrascale+ |

## 3. Interface

The accelerator exposes an **AXI4 Lite-32 bit @100MHz for Control transfer + AXI4 Stream-512 bit @250MHz for Data transfer** interface.

### Why AXI and not SPI or APB

| Option | Reason not chosen |
|---|---|
| **SPI** | Single-bit serial — far too slow to transfer 512-wide activation rows at inference throughput |
| **APB** | No burst support; loading 4,096 weight tiles via single-cycle transactions is impractical |
| **Wishbone** | Minimal Xilinx/AMD ecosystem support; no native DMA integration |
| **AXI4-Lite + AXI4-Stream** | Burst weight DMA, native PL-PS interconnect on Xilinx Ultrascale+, backpressure on data streams |

### AXI4-Lite control plane

Register map (32-bit, word-addressed):

| Offset | Register | Description |
|---|---|---|
| `0x00` | `CTRL` | Bit 0 = `start`, bit 1 = `use_mask`, bit 2 = `busy` (RO), bit 3 = `done` (RO, W1C) |
| `0x04` | `WEIGHT_SEL` | Matrix: 0=W_Q 1=W_K 2=W_V 3=W_pos 4=W_O |
| `0x08` | `WEIGHT_TILE_ADDR` | `{weight_tr[7:0], weight_tc[7:0]}` |
| `0x0C` | `BIAS_SEL` | 0 = u_bias, 1 = v_bias |
| `0x10` | `FSM_STATE` | Read-only, waveform debug |
| `0x14` | `SEQ_LEN_CFG` | Runtime sequence length (≤ `SEQ_LEN` parameter) |

Weight tiles written via burst to aperture `0x1000`.

### AXI4-Stream data plane

```
S_AXIS_X  : 512-bit  input activations   TUSER = row index
S_AXIS_P  : 512-bit  positional embeddings  TUSER = row index
M_AXIS_Y  : 512-bit  output              TLAST on final row  
```

### Calculating required Bandwidth from Arithmetic Intensity
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

### Calculating Interface Bandwidth
**Data Pane**  
Bus width = 512 bits = 64 bytes per transfer
Clock frequency = 250 MHz  
Data BW = Bytes per transfer * clock freq  
      = 64 \* 250 MHz  
      = 16 GB/s

Required BW = 8.19 GB/s  
Interface rated BW for data:= 16 GB/s  