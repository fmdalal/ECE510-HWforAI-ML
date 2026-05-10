// ============================================================
// File   : codefest/cf06/hdl/crossbar_tb.sv
// Desc   : Testbench for serial accumulator crossbar MAC
//          Weights: [[1,-1,1,-1],[1,1,-1,-1],[-1,1,1,-1],[-1,-1,-1,1]]
//          Input  : [10, 20, 30, 40]
// Hand calculated:
//   out[0] = +10 +20 -30 -40 = -40
//   out[1] = -10 +20 +30 -40 =   0
//   out[2] = +10 -20 +30 -40 = -20
//   out[3] = -10 -20 -30 +40 = -20
// ============================================================
`timescale 1ns/1ps

module crossbar_tb;

    logic        clk, rst_n;
    logic [31:0] in_vec;
    logic [15:0] weight;
    logic [39:0] out_vec;

    crossbar_mac dut (
        .clk(clk), .rst_n(rst_n),
        .in_vec(in_vec), .weight(weight), .out_vec(out_vec)
    );

    // Signed output aliases
    wire signed [9:0] out0 = out_vec[9:0];
    wire signed [9:0] out1 = out_vec[19:10];
    wire signed [9:0] out2 = out_vec[29:20];
    wire signed [9:0] out3 = out_vec[39:30];

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_cnt, fail_cnt;

    task check_outputs;
        input signed [9:0] e0,e1,e2,e3;
        input [255:0] label;
        begin
            $display("  %s", label);
            $display("  out    = [%4d, %4d, %4d, %4d]", out0,out1,out2,out3);
            $display("  expect = [%4d, %4d, %4d, %4d]", e0,e1,e2,e3);
            if (out0===e0 && out1===e1 && out2===e2 && out3===e3) begin
                $display("  --> PASS"); pass_cnt=pass_cnt+1;
            end else begin
                $display("  --> FAIL"); fail_cnt=fail_cnt+1;
            end
        end
    endtask

    initial begin
        pass_cnt=0; fail_cnt=0;

        $display("=================================================");
        $display("  crossbar_tb: Serial MAC Accumulator Testbench");
        $display("=================================================");

        // Weights packed: 16'b1000_0110_0011_0101
        // row0(i=0): j0=1,j1=0,j2=1,j3=0 → bits 3:0  = 4'b0101
        // row1(i=1): j0=1,j1=1,j2=0,j3=0 → bits 7:4  = 4'b0011
        // row2(i=2): j0=0,j1=1,j2=1,j3=0 → bits 11:8 = 4'b0110
        // row3(i=3): j0=0,j1=0,j2=0,j3=1 → bits 15:12= 4'b1000
        in_vec = {8'd40, 8'd30, 8'd20, 8'd10};
        weight = 16'b1000_0110_0011_0101;

        // Reset
        rst_n = 0;
        @(posedge clk); #1;
        @(posedge clk); #1;
        rst_n = 1;

        $display("\n--- Cycle-by-cycle accumulation ---");

        // Cycle 1: row0 processed — in[0]=10, weights [+1,-1,+1,-1]
        // acc = [+10, -10, +10, -10]
        @(posedge clk); #1;
        $display("\n[Cycle 1] row0 processed: in[0]=10");
        check_outputs(10, -10, 10, -10, "acc += weight[0][j] * in[0]");

        // Cycle 2: row1 processed — in[1]=20, weights [+1,+1,-1,-1]
        // acc = [+10+20, -10+20, +10-20, -10-20] = [30, 10, -10, -30]
        @(posedge clk); #1;
        $display("\n[Cycle 2] row1 processed: in[1]=20");
        check_outputs(30, 10, -10, -30, "acc += weight[1][j] * in[1]");

        // Cycle 3: row2 processed — in[2]=30, weights [-1,+1,+1,-1]
        // acc = [30-30, 10+30, -10+30, -30-30] = [0, 40, 20, -60]
        @(posedge clk); #1;
        $display("\n[Cycle 3] row2 processed: in[2]=30");
        check_outputs(0, 40, 20, -60, "acc += weight[2][j] * in[2]");

        // Cycle 4: row3 processed — in[3]=40, weights [-1,-1,-1,+1]
        // acc = [0-40, 40-40, 20-40, -60+40] = [-40, 0, -20, -20]
        @(posedge clk); #1;
        $display("\n[Cycle 4] row3 processed: in[3]=40 --> FINAL RESULT");
        check_outputs(-40, 0, -20, -20, "acc += weight[3][j] * in[3]");

        $display("\n=================================================");
        $display("  Results: %0d passed, %0d failed", pass_cnt, fail_cnt);
        if (fail_cnt==0)
            $display("  ALL TESTS PASSED - matches hand calculation");
        else
            $display("  SOME TESTS FAILED");
        $display("=================================================");
        $finish;
    end

    initial begin
        $dumpfile("crossbar_tb.vcd");
        $dumpvars(0, crossbar_tb);
    end

endmodule
