# ECE510-HWforAI-ML
## M4 — Hardware Accelerator Submission
**Name:**                       Fatema Chikani  
**Course:**                     ECE510 Spring 2026  
**Project topic:**    9-Chiplet Mixed Precision Multi-Head Self-Attention (MHSA) Hardware Accelerator for Conformer-based Automatic Speech Recognition



This repository contains the complete M4 submission for the **9-Chiplet Mixed Precsion Multi-Head Self-Attention (MHSA) Hardware Accelerator**, targeting the Conformer speech recognition architecture ([sooftware/conformer](https://github.com/sooftware/conformer)). The accelerator implements the full MHSA pipeline across nine specialised chiplets — QKV projection, eight parallel attention heads, softmax, and output projection — connected via UCIe 1.0 inter-chiplet links and a weight-stationary dataflow that eliminates weight reload overhead across token tiles. RTL was written in SystemVerilog, verified through QuestaSim 2021.3\_1 functional simulation (PASS, 0/1,024 BF16 mismatches within 2 ULP), and synthesised using Cadence Genus 19.12-s121\_1 on GPDK045 45 nm. The measured result is a **9.0× per-token latency speedup** over the Intel Core Ultra 7 256V software baseline (18.14 µs → 2.026 µs/token) and a **16× energy reduction** (154 mJ → 9.35 mJ per inference, SA kernel basis).

→ **[project/m4/README.md](project/m4/README.md)** — full file catalogue, one line per file with description and checklist mapping  
→ **[project/m4/report/design_justification_report.docx](project/m4/report/design_justification_report.docx)** — 9-section design justification report (Problem/Motivation, Roofline, Precision, Dataflow, Hardware Interface, Verification, Synthesis Results, Benchmark, What Did Not Work)

---

## Repository Structure

```
repo/
├── README.md                        ← this file
├── project/
│   ├── heilmeier.md                 ← from M1
│   ├── m1/                          ← from M1
│   ├── m2/                          ← from M2
│   ├── m3/                          ← from M3
│   └── m4/
│       ├── README.md                ← M4 file catalogue
│       ├── rtl/                     ← final RTL source (9 .sv files)
│       ├── tb/                      ← testbenches (tb_top.sv, cocotb M2)
│       ├── sim/                     ← simulation outputs (sim_run.log, waveform)
│       ├── synth/                   ← synthesis results (timing, area, power .rpt)
│       ├── bench/                   ← benchmark.md + benchmark_raw_data.csv
│       └── report/                  ← design_justification_report.docx + figures/
└── codefest/                        ← from earlier weeks
```

