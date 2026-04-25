// =============================================================================
// Module      : mac_tb
// Description : Testbench for INT8 MAC unit
//               Test sequence:
//               1. Apply [a=3, b=4] for 3 cycles
//               2. Assert rst (synchronous active-high)
//               3. Apply [a=-5, b=2] for 2 cycles
// =============================================================================

`timescale 1ns/1ps

module mac_tb;

    // -------------------------------------------------------------------------
    // DUT signal declarations
    // -------------------------------------------------------------------------
    logic        clk;
    logic        rst;
    logic signed [7:0]  a;
    logic signed [7:0]  b;
    logic signed [31:0] out;

    // -------------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------------
    mac dut (
        .clk (clk),
        .rst (rst),
        .a   (a),
        .b   (b),
        .out (out)
    );

    // -------------------------------------------------------------------------
    // Clock generation: 10ns period (100MHz)
    // -------------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // Task: wait for rising edge then display output
    // -------------------------------------------------------------------------
    task tick(input string label);
        @(posedge clk);
        #1; // small delay to let output settle
        $display("[%0t] %s | a=%0d, b=%0d, rst=%0b | out=%0d", 
                  $time, label, a, b, rst, out);
    endtask

    // -------------------------------------------------------------------------
    // Stimulus
    // -------------------------------------------------------------------------
    initial begin
        // Dump waveform
        $dumpfile("mac_tb.vcd");
        $dumpvars(0, mac_tb);

        $display("=============================================================");
        $display("                  MAC Unit Testbench                        ");
        $display("=============================================================");

        // -- Initial reset --
        rst = 1;
        a   = 8'sd0;
        b   = 8'sd0;
        tick("RESET      ");

        rst = 0;

        // -- Phase 1: a=3, b=4 for 3 cycles --
        // Expected accumulation: 12 → 24 → 36
        $display("-------------------------------------------------------------");
        $display(" Phase 1: a=3, b=4 for 3 cycles (expected: 12, 24, 36)");
        $display("-------------------------------------------------------------");
        a = 8'sd3;
        b = 8'sd4;
        tick("CYCLE 1    ");
        assert (out === 32'sd12) else $error("FAIL: expected 12, got %0d", out);

        tick("CYCLE 2    ");
        assert (out === 32'sd24) else $error("FAIL: expected 24, got %0d", out);

        tick("CYCLE 3    ");
        assert (out === 32'sd36) else $error("FAIL: expected 36, got %0d", out);

        // -- Phase 2: assert rst --
        $display("-------------------------------------------------------------");
        $display(" Phase 2: Assert rst (expected: out=0)");
        $display("-------------------------------------------------------------");
        rst = 1;
        a   = 8'sd0;
        b   = 8'sd0;
        tick("RST ASSERT ");
        assert (out === 32'sd0) else $error("FAIL: expected 0 after reset, got %0d", out);

        rst = 0;

        // -- Phase 3: a=-5, b=2 for 2 cycles --
        // Expected accumulation: -10 → -20
        $display("-------------------------------------------------------------");
        $display(" Phase 3: a=-5, b=2 for 2 cycles (expected: -10, -20)");
        $display("-------------------------------------------------------------");
        a = -8'sd5;
        b =  8'sd2;
        tick("CYCLE 1    ");
        assert (out === -32'sd10) else $error("FAIL: expected -10, got %0d", out);

        tick("CYCLE 2    ");
        assert (out === -32'sd20) else $error("FAIL: expected -20, got %0d", out);

        $display("=============================================================");
        $display(" Simulation complete.");
        $display("=============================================================");

        $finish;
    end

endmodule