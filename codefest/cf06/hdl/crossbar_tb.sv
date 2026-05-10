// ============================================================
// File   : codefest/cf06/hdl/crossbar_tb.sv
// Test   : 4x4 binary-weight crossbar MAC
// Weights: [[+1,-1,+1,-1],[+1,+1,-1,-1],[-1,+1,+1,-1],[-1,-1,-1,+1]]
// Inputs : [10, 20, 30, 40]
// Expected outputs (hand-calculated):
//   out[0] = +10 +20 -30 -40 = -40
//   out[1] = -10 +20 +30 -40 =   0
//   out[2] = +10 -20 +30 -40 = -20
//   out[3] = -10 -20 -30 +40 = -20
// ============================================================
`timescale 1ns/1ps

module crossbar_tb;

    // ---- DUT ports ----
    logic        clk;
    logic        rst_n;
    logic [31:0] in_vec;    // {in[3],in[2],in[1],in[0]} each 8-bit signed
    logic [15:0] weight;    // weight[i*4+j], bit=1 → +1, bit=0 → -1
    logic [39:0] out_vec;   // {out[3],out[2],out[1],out[0]} each 10-bit signed

    // ---- Instantiate DUT ----
    crossbar_mac dut (
        .clk    (clk),
        .rst_n  (rst_n),
        .in_vec (in_vec),
        .weight (weight),
        .out_vec(out_vec)
    );

    // ---- Signed output aliases ----
    wire signed [9:0] out0 = out_vec[9:0];
    wire signed [9:0] out1 = out_vec[19:10];
    wire signed [9:0] out2 = out_vec[29:20];
    wire signed [9:0] out3 = out_vec[39:30];

    // ---- Clock: 10 ns period ----
    initial clk = 0;
    always #5 clk = ~clk;

    // ---- Pass/fail counter ----
    integer pass_cnt, fail_cnt;

    // ---- Check task ----
    task check_outputs;
        input signed [9:0] e0, e1, e2, e3;
        begin
            $display("  out    = [%4d, %4d, %4d, %4d]", out0, out1, out2, out3);
            $display("  expect = [%4d, %4d, %4d, %4d]", e0,   e1,   e2,   e3);
            if (out0===e0 && out1===e1 && out2===e2 && out3===e3) begin
                $display("  --> PASS");
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("  --> FAIL");
                if (out0!==e0) $display("    out[0]: got %0d, expected %0d", out0, e0);
                if (out1!==e1) $display("    out[1]: got %0d, expected %0d", out1, e1);
                if (out2!==e2) $display("    out[2]: got %0d, expected %0d", out2, e2);
                if (out3!==e3) $display("    out[3]: got %0d, expected %0d", out3, e3);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    // ---- Main test sequence ----
    initial begin
        pass_cnt = 0;
        fail_cnt = 0;

        $display("=================================================");
        $display("  crossbar_tb: 4x4 Binary-Weight MAC Testbench");
        $display("=================================================");

        // Initialise and reset
        rst_n  = 0;
        in_vec = 32'h00000000;
        weight = 16'hFFFF;
        @(posedge clk); #1;
        @(posedge clk); #1;
        rst_n = 1;
        @(posedge clk); #1;

        // -------------------------------------------------------
        // Test 1 — Primary test (matches assignment specification)
        // Weights: [[+1,-1,+1,-1],[+1,+1,-1,-1],[-1,+1,+1,-1],[-1,-1,-1,+1]]
        // Packed weight[i*4+j]:
        //   row0 (i=0): j0=1,j1=0,j2=1,j3=0  → bits  3:0  = 4'b0101
        //   row1 (i=1): j0=1,j1=1,j2=0,j3=0  → bits  7:4  = 4'b0011
        //   row2 (i=2): j0=0,j1=1,j2=1,j3=0  → bits 11:8  = 4'b0110
        //   row3 (i=3): j0=0,j1=0,j2=0,j3=1  → bits 15:12 = 4'b1000
        //   Combined: 16'b1000_0110_0011_0101
        // Hand-calculated outputs:
        //   out[0] = +10 +20 -30 -40 = -40
        //   out[1] = -10 +20 +30 -40 =   0
        //   out[2] = +10 -20 +30 -40 = -20
        //   out[3] = -10 -20 -30 +40 = -20
        // -------------------------------------------------------
        $display("\n[Test 1] Spec weights, in=[10,20,30,40]");
        $display("  Weight matrix:");
        $display("         col0 col1 col2 col3");
        $display("  row0: [ +1,  -1,  +1,  -1 ]");
        $display("  row1: [ +1,  +1,  -1,  -1 ]");
        $display("  row2: [ -1,  +1,  +1,  -1 ]");
        $display("  row3: [ -1,  -1,  -1,  +1 ]");
        in_vec = {8'd40, 8'd30, 8'd20, 8'd10};   // {in3,in2,in1,in0}
        weight = 16'b1000_0110_0011_0101;
        @(posedge clk); #1;
        check_outputs(-40, 0, -20, -20);

        // -------------------------------------------------------
        // Test 2 — All weights +1, uniform inputs
        // out[j] = 1+1+1+1 = 4 for all j
        // -------------------------------------------------------
        $display("\n[Test 2] All +1 weights, in=[1,1,1,1]");
        in_vec = {8'd1, 8'd1, 8'd1, 8'd1};
        weight = 16'hFFFF;
        @(posedge clk); #1;
        check_outputs(4, 4, 4, 4);

        // -------------------------------------------------------
        // Test 3 — All weights -1, uniform inputs
        // out[j] = -1-1-1-1 = -4 for all j
        // -------------------------------------------------------
        $display("\n[Test 3] All -1 weights, in=[1,1,1,1]");
        weight = 16'h0000;
        @(posedge clk); #1;
        check_outputs(-4, -4, -4, -4);

        // -------------------------------------------------------
        // Test 4 — Diagonal +1, rest -1, in=[10,20,30,40]
        // out[0]=+10-20-30-40=-80
        // out[1]=-10+20-30-40=-60
        // out[2]=-10-20+30-40=-40
        // out[3]=-10-20-30+40=-20
        // weight[i*4+j]=1 only when i==j: bits 0,5,10,15 → 16'h8421
        // -------------------------------------------------------
        $display("\n[Test 4] Diagonal +1, rest -1, in=[10,20,30,40]");
        in_vec = {8'd40, 8'd30, 8'd20, 8'd10};
        weight = 16'h8421;
        @(posedge clk); #1;
        check_outputs(-80, -60, -40, -20);

        // -------------------------------------------------------
        // Test 5 — Negative inputs, all +1 weights
        // out[j] = -5-10-15-20 = -50 for all j
        // -------------------------------------------------------
        $display("\n[Test 5] All +1 weights, in=[-5,-10,-15,-20]");
        in_vec = {8'(-20), 8'(-15), 8'(-10), 8'(-5)};
        weight = 16'hFFFF;
        @(posedge clk); #1;
        check_outputs(-50, -50, -50, -50);

        // -------------------------------------------------------
        // Summary
        // -------------------------------------------------------
        $display("\n=================================================");
        $display("  Results: %0d passed, %0d failed", pass_cnt, fail_cnt);
        if (fail_cnt == 0)
            $display("  ALL TESTS PASSED");
        else
            $display("  SOME TESTS FAILED");
        $display("=================================================");

        $finish;
    end

    // ---- Waveform dump ----
    initial begin
        $dumpfile("crossbar_tb.vcd");
        $dumpvars(0, crossbar_tb);
    end

endmodule
