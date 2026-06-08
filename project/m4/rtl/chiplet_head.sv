// =============================================================================
// chiplet_head.sv  —  Chiplets 1..8  [optimised: weight-stationary + staggered A]
// =============================================================================
// Changes from original
// ----------------------
//  1. Tile-streaming FSM: iterates k=0..TILE-1, presenting:
//       a_col_in[i] = A[i][k]  (full column of Q or probs, not just col 0)
//       b_row_in[j] = B[k][j]  (full row of Kt or V)
//     The systolic array handles all staggering internally.
//
//  2. B (weights/Kt/V) are captured into local registers when rxb_valid.
//     They are then streamed one row per cycle (weight-stationary: loaded
//     once on-chip, never reloaded during the multiply).
//
//  3. acc_clear pulsed once before each new tile multiply.
//
//  4. Stage-2 transpose: done at capture time (rxb_valid), not per-cycle.
//     kt_tile[i][j] = rxb_tile[j][i] — full matrix, captured in 1 cycle.
//
//  5. Scale applied combinationally on sa_c_out at valid_out, captured in TX state.
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module chiplet_head #(
    parameter int HEAD_ID   = 0,
    parameter int D_HEAD    = 64,
    parameter int TILE      = 64,
    parameter int K_DIM     = 64,
    parameter int SEQ_TILE  = 64
)(
    input  wire        clk_core,
    input  wire        clk_link,
    input  wire        rst_n,
    input  wire        cfg_mode,
    input  wire [7:0]  cfg_num_tiles,
    input  wire        cfg_start,
    output logic       cfg_done,
    output logic [3:0] chiplet_id,

    input  wire [511:0] rxa_bump_data,
    input  wire         rxa_bump_valid,
    output wire         rxa_bump_credit,

    input  wire [511:0] rxb_bump_data,
    input  wire         rxb_bump_valid,
    output wire         rxb_bump_credit,

    output wire [511:0] tx_bump_data,
    output wire         tx_bump_valid,
    input  wire         tx_bump_credit,

    input  wire [15:0]  scale_factor
);
    assign chiplet_id = HEAD_ID[3:0] + 4'd1;

    // =========================================================================
    // UCIe RX
    // =========================================================================
    logic        rxa_valid;
    logic [15:0] rxa_tile [TILE][TILE];
    ucie_rx #(.TILE_DIM(TILE)) u_rxa (
        .clk_core(clk_core),.clk_link(clk_link),.rst_n(rst_n),
        .bump_data(rxa_bump_data),.bump_valid(rxa_bump_valid),.bump_credit(rxa_bump_credit),
        .rx_valid(rxa_valid),.rx_src_id(),.rx_tile(rxa_tile),.rx_ready(1'b1),
        .rx_crc_err(),.rx_seq_err()
    );

    logic        rxb_valid;
    logic [15:0] rxb_tile [TILE][TILE];
    ucie_rx #(.TILE_DIM(TILE)) u_rxb (
        .clk_core(clk_core),.clk_link(clk_link),.rst_n(rst_n),
        .bump_data(rxb_bump_data),.bump_valid(rxb_bump_valid),.bump_credit(rxb_bump_credit),
        .rx_valid(rxb_valid),.rx_src_id(),.rx_tile(rxb_tile),.rx_ready(1'b1),
        .rx_crc_err(),.rx_seq_err()
    );

    // =========================================================================
    // Tile capture registers
    // A tile: Q (stage2) or softmax probs (stage4)
    // B tile: Kt (stage2, transposed at capture) or V (stage4, pass-through)
    // =========================================================================
    logic [15:0] a_tile [TILE][TILE];
    logic [15:0] b_tile [TILE][TILE];   // b_tile[k][j] = row k, col j of B

    always_ff @(posedge clk_core or negedge rst_n) begin : a_cap
        if (!rst_n) for (int i=0;i<TILE;i++) for (int j=0;j<TILE;j++) a_tile[i][j]<=16'h0;
        else if (rxa_valid) for (int i=0;i<TILE;i++) for (int j=0;j<TILE;j++) a_tile[i][j]<=rxa_tile[i][j];
    end

    always_ff @(posedge clk_core or negedge rst_n) begin : b_cap
        if (!rst_n) for (int i=0;i<TILE;i++) for (int j=0;j<TILE;j++) b_tile[i][j]<=16'h0;
        else if (rxb_valid)
            for (int i=0;i<TILE;i++)
                for (int j=0;j<TILE;j++)
                    // stage2: transpose K → b_tile[k][j]=K[j][k]=Kt[k][j]
                    // stage4: pass V through
                    b_tile[i][j] <= cfg_mode ? rxb_tile[i][j] : rxb_tile[j][i];
    end

    // =========================================================================
    // Tile-streaming FSM
    // =========================================================================
    typedef enum logic [2:0] {S_IDLE,S_WAIT,S_STREAM,S_DRAIN,S_TX} fsm_t;
    fsm_t                        state;
    logic [$clog2(TILE)-1:0]     k_cnt;
    logic                        a_cap_done, b_cap_done;

    always_ff @(posedge clk_core or negedge rst_n) begin : cap_track
        if (!rst_n) begin a_cap_done<=1'b0; b_cap_done<=1'b0; end
        else begin
            if (!cfg_start) begin a_cap_done<=1'b0; b_cap_done<=1'b0; end
            else begin
                if (rxa_valid) a_cap_done<=1'b1;
                if (rxb_valid) b_cap_done<=1'b1;
                if (state==S_STREAM && k_cnt==TILE-1) begin
                    a_cap_done<=1'b0; b_cap_done<=1'b0;
                end
            end
        end
    end

    logic        sa_data_in, sa_acc_clear, sa_flush;
    logic [15:0] sa_a_col [TILE];
    logic [15:0] sa_b_row [TILE];

    // Combinational slice selectors
    always_comb begin
        for (int i=0;i<TILE;i++) sa_a_col[i] = a_tile[i][k_cnt];  // col k of A
        for (int j=0;j<TILE;j++) sa_b_row[j] = b_tile[k_cnt][j];  // row k of B
    end

    wire [15:0] sa_c_out [TILE][TILE];
    wire        sa_valid;

    always_ff @(posedge clk_core or negedge rst_n) begin : fsm
        if (!rst_n) begin
            state<=S_IDLE; k_cnt<='0;
            sa_data_in<=1'b0; sa_acc_clear<=1'b0; sa_flush<=1'b0; cfg_done<=1'b0;
        end else begin
            sa_data_in<=1'b0; sa_acc_clear<=1'b0; sa_flush<=1'b0; cfg_done<=1'b0;
            case (state)
                S_IDLE: if (cfg_start) begin k_cnt<='0; state<=S_WAIT; end
                S_WAIT: begin
                    if (!cfg_start) state<=S_IDLE;
                    else if (a_cap_done & b_cap_done) begin
                        sa_acc_clear<=1'b1; k_cnt<='0; state<=S_STREAM;
                    end
                end
                S_STREAM: begin
                    sa_data_in<=1'b1;
                    if (k_cnt == (TILE-1)) begin
                        sa_flush<=1'b1; k_cnt<='0; state<=S_DRAIN;
                    end else k_cnt<=k_cnt+1'b1;
                end
                S_DRAIN: if (sa_valid) state<=S_TX;
                S_TX: begin cfg_done<=cfg_mode; state<=S_WAIT; end
                default: state<=S_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Systolic array
    // =========================================================================
    systolic_array #(.M(TILE),.N(TILE),.K(K_DIM)) sa_main (
        .clk_core(clk_core),.rst_n(rst_n),
        .data_in(sa_data_in),.acc_clear(sa_acc_clear),.flush(sa_flush),
        .a_col_in(sa_a_col),.b_row_in(sa_b_row),
        .c_out(sa_c_out),.valid_out(sa_valid)
    );

    // =========================================================================
    // Stage-2 scale: 1/sqrt(D_HEAD) applied combinationally on sa_c_out
    // =========================================================================
    wire [15:0] scaled_out [TILE][TILE];
    genvar si,sj;
    generate
        for (si=0;si<TILE;si++) begin : sc_row
            for (sj=0;sj<TILE;sj++) begin : sc_col
                wire [31:0] sc_fp32;
                fp32_mul u_scl (.a(sa_c_out[si][sj]),.b(scale_factor),.result(sc_fp32));
                wire rup = sc_fp32[15]&(|sc_fp32[14:0]|sc_fp32[16]);
                assign scaled_out[si][sj] = sc_fp32[31:16]+{15'h0,rup};
            end
        end
    endgenerate

    // =========================================================================
    // Output register + UCIe TX
    // =========================================================================
    logic        tx_valid_i;
    logic [15:0] tx_tile [TILE][TILE];
    logic [3:0]  tx_dst;

    always_ff @(posedge clk_core or negedge rst_n) begin : tx_reg
        if (!rst_n) begin tx_valid_i<=1'b0; tx_dst<=4'd0; end
        else begin
            tx_valid_i<=1'b0;
            if (state==S_TX) begin
                tx_valid_i<=1'b1;
                if (~cfg_mode) begin
                    tx_dst<=4'd9;
                    for (int i=0;i<TILE;i++) for (int j=0;j<TILE;j++) tx_tile[i][j]<=scaled_out[i][j];
                end else begin
                    tx_dst<=4'd0;
                    for (int i=0;i<TILE;i++) for (int j=0;j<TILE;j++) tx_tile[i][j]<=sa_c_out[i][j];
                end
            end
        end
    end

    ucie_tx #(.TILE_DIM(TILE)) u_tx (
        .clk_core(clk_core),.clk_link(clk_link),.rst_n(rst_n),
        .tx_valid(tx_valid_i),.tx_src_id(HEAD_ID[3:0]+4'd1),.tx_dst_id(tx_dst),
        .tx_tile(tx_tile),.tx_ready(),
        .bump_data(tx_bump_data),.bump_valid(tx_bump_valid),.bump_credit(tx_bump_credit)
    );

endmodule

`default_nettype wire
// =============================================================================
// End of chiplet_head.sv  (optimised)
// =============================================================================
