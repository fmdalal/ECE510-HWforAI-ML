// ============================================================
// File   : codefest/cf06/hdl/crossbar_mac.sv
// Desc   : 4x4 binary-weight crossbar MAC unit
//          Each clock cycle accumulates one input row:
//          cycle 0: out[j] += weight[0][j] * in[0]
//          cycle 1: out[j] += weight[1][j] * in[1]
//          cycle 2: out[j] += weight[2][j] * in[2]
//          cycle 3: out[j] += weight[3][j] * in[3]
//          After 4 cycles: out[j] = Σ_i weight[i][j] * in[i]
// Inputs : in_vec  — 4x 8-bit signed, packed [31:0]
//          weight  — 4x4 bits packed [15:0], weight[i*4+j]
//                    1 = +1, 0 = -1
// Outputs: out_vec — 4x 10-bit signed, packed [39:0]
// ============================================================
module crossbar_mac (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [31:0] in_vec,   // {in[3],in[2],in[1],in[0]}
    input  logic [15:0] weight,   // weight[i*4+j]
    output logic [39:0] out_vec   // {out[3],out[2],out[1],out[0]}
);

    // Unpack inputs
    wire signed [7:0] inv [0:3];
    assign inv[0] = in_vec[7:0];
    assign inv[1] = in_vec[15:8];
    assign inv[2] = in_vec[23:16];
    assign inv[3] = in_vec[31:24];

    // Accumulator registers — 4 outputs each 10-bit signed
    logic signed [9:0] acc [0:3];

    // Row counter — tracks which input row is being processed
    logic [1:0] row_cnt;

    // Signed output aliases
    assign out_vec[9:0]   = acc[0];
    assign out_vec[19:10] = acc[1];
    assign out_vec[29:20] = acc[2];
    assign out_vec[39:30] = acc[3];

    integer jj;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset accumulators and counter
            for (jj=0; jj<4; jj++)
                acc[jj] <= 10'sd0;
            row_cnt <= 2'd0;
        end else begin
            // Each cycle: accumulate current row into all 4 outputs
            for (jj=0; jj<4; jj++) begin
                if (weight[row_cnt*4+jj])
                    acc[jj] <= acc[jj] + 10'(signed'(inv[row_cnt]));
                else
                    acc[jj] <= acc[jj] - 10'(signed'(inv[row_cnt]));
            end
            row_cnt <= row_cnt + 1;
        end
    end

endmodule
