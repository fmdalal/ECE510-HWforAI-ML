// Module: mac
// Description: 8-bit signed Multiply-Accumulate unit with 32-bit output
module mac (
    input  logic              clk,  // 1-bit clock
    input  logic              rst,  // 1-bit synchronous active-high reset
    input  logic signed [7:0] a,    // 8-bit signed input A
    input  logic signed [7:0] b,    // 8-bit signed input B
    output logic signed [31:0] out  // 32-bit signed output (accumulator)
);

    // Internal signal for the product of a * b
    // 8-bit * 8-bit results in a 16-bit signed value
    logic signed [15:0] product;

    assign product = a * b;

    always_ff @(posedge clk) begin
        if (rst) begin
            out <= 32'sd0;
        end else begin
            // Accumulate the product into the 32-bit output register
            out <= out + 32'(product);
        end
    end

endmodule