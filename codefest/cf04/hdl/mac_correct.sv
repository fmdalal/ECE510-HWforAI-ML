// =============================================================================
// Module      : mac
// Description : INT8 Multiply-Accumulate Unit (SystemVerilog)
//               Computes: out <= out + (a * b)
// Ports       : clk - 1-bit clock
//               rst - 1-bit synchronous active-high reset
//               a   - 8-bit signed operand
//               b   - 8-bit signed operand
//               out - 32-bit signed accumulated result
// Reset       : Synchronous, active-high
// Synthesis   : Fully synthesizable
// =============================================================================

module mac (
    input  logic        clk,  // Clock
    input  logic        rst,  // Synchronous active-high reset
    input  logic signed [7:0]  a,   // 8-bit signed input A
    input  logic signed [7:0]  b,   // 8-bit signed input B
    output logic signed [31:0] out  // 32-bit signed accumulated output
);

    // -------------------------------------------------------------------------
    // Internal: 16-bit signed product of a * b
    // 8-bit x 8-bit signed multiply produces a 16-bit full-precision result
    // -------------------------------------------------------------------------
    logic signed [15:0] product;

    assign product = a * b;

    // -------------------------------------------------------------------------
    // Synchronous MAC register
    //   Reset  : clear accumulator to 0
    //   Normal : sign-extend product to 32 bits, accumulate into out
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            out <= '0;
        end else begin
            out <= out + 32'(signed'(product));  // sign-extend 16→32 bits
        end
    end

endmodule
