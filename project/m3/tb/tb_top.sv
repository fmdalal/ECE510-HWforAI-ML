// =============================================================================
// tb_top.sv  --  Conformer Accelerator Co-Simulation Testbench
// Questa 2021.3_1 / Icarus Verilog 12
// =============================================================================
`timescale 1ns/1ps
`default_nettype none

// =============================================================================
// Behavioural DUT -- AXI4-Lite CSR + AXI4-Stream + BF16 softmax
// Implements the identical host protocol as the RTL hierarchy.
// =============================================================================
module dut #(
    parameter int TILE_DIM = 8,
    parameter int TDATA_W  = 512
)(
    input  wire        clk_axi, rst_n,
    // AXI4-Lite slave
    input  wire [11:0] s_axil_awaddr,
    input  wire        s_axil_awvalid,
    output reg         s_axil_awready,
    input  wire [31:0] s_axil_wdata,
    input  wire        s_axil_wvalid,
    output reg         s_axil_wready,
    output reg  [1:0]  s_axil_bresp,
    output reg         s_axil_bvalid,
    input  wire        s_axil_bready,
    input  wire [11:0] s_axil_araddr,
    input  wire        s_axil_arvalid,
    output reg         s_axil_arready,
    output reg  [31:0] s_axil_rdata,
    output reg  [1:0]  s_axil_rresp,
    output reg         s_axil_rvalid,
    input  wire        s_axil_rready,
    // AXI4-Stream slave
    input  wire [TDATA_W-1:0] s_axis_tdata,
    input  wire               s_axis_tvalid,
    output reg                s_axis_tready,
    input  wire               s_axis_tlast,
    // AXI4-Stream master
    output reg  [TDATA_W-1:0] m_axis_tdata,
    output reg                m_axis_tvalid,
    input  wire               m_axis_tready,
    output reg                m_axis_tlast
);
    reg        cfg_start_r, r_busy, r_done;
    reg [15:0] in_tile [0:TILE_DIM*TILE_DIM-1];
    reg [15:0] out_tile[0:TILE_DIM*TILE_DIM-1];
    integer    beat_in, ws, rs, ps, pd, beat_out;
    reg [11:0] waddr_l;

    // Write FSM: WR_IDLE(0) ? WR_DATA(1) ? WR_RESP(2)
    always_ff @(posedge clk_axi or negedge rst_n) begin
        if (!rst_n) begin
            ws<=0; s_axil_awready<=0; s_axil_wready<=0;
            s_axil_bvalid<=0; s_axil_bresp<=0;
            cfg_start_r<=0;
        end else begin
            cfg_start_r <= 0;
            case (ws)
                0: begin s_axil_awready<=1;
                         if (s_axil_awvalid) begin
                             s_axil_awready<=0; waddr_l<=s_axil_awaddr; ws<=1;
                         end end
                1: begin s_axil_wready<=1;
                         if (s_axil_wvalid) begin
                             s_axil_wready<=0;
                             if (waddr_l==12'h000) cfg_start_r <= s_axil_wdata[0];
                             ws<=2;
                         end end
                2: begin s_axil_bvalid<=1; s_axil_bresp<=0;
                         if (s_axil_bready) begin s_axil_bvalid<=0; ws<=0; end end
            endcase
        end
    end

    // Read FSM
    always_ff @(posedge clk_axi or negedge rst_n) begin
        if (!rst_n) begin rs<=0; s_axil_arready<=0; s_axil_rvalid<=0; end
        else case (rs)
            0: begin s_axil_arready<=1; s_axil_rvalid<=0;
                     if (s_axil_arvalid) begin s_axil_arready<=0; rs<=1; end end
            1: begin s_axil_rvalid<=1; s_axil_rresp<=0;
                     s_axil_rdata <= (s_axil_araddr==12'h004) ?
                                     {30'b0,r_done,r_busy} : 32'hDEAD_BEEF;
                     rs<=2; end
            2: begin if (s_axil_rready) begin s_axil_rvalid<=0; rs<=0; end end
        endcase
    end

    // Stream in
    always_ff @(posedge clk_axi or negedge rst_n) begin
        if (!rst_n) begin s_axis_tready<=1; beat_in<=0; end
        else if (s_axis_tvalid && s_axis_tready) begin
            for (int w=0; w<32; w++) begin
                int idx; idx = beat_in*32+w;
                if (idx < TILE_DIM*TILE_DIM)
                    in_tile[idx] <= s_axis_tdata[w*16 +: 16];
            end
            if (s_axis_tlast) begin beat_in<=0; r_busy<=1; end
            else               beat_in <= beat_in+1;
        end
    end

    // Softmax compute + stream out
    real v[0:TILE_DIM-1]; real mx, sm;

    function real b2r(input [15:0] h);
        real sv,mv; integer ev;
        sv=(h[15]?-1.0:1.0); ev=h[14:7]-127; mv=1.0+$itor(h[6:0])/128.0;
        b2r=(h[14:7]==0)?0.0:sv*mv*(2.0**ev);
    endfunction

    function [15:0] r2b(input real x);
        real av,mf; integer ei,mi;
        if (x<=0.0) begin r2b=16'h0000; end
        else begin
            av=x; ei=0;
            while(av>=2.0)begin av=av/2.0; ei=ei+1; end
            while(av<1.0) begin av=av*2.0; ei=ei-1; end
            mf=av-1.0; mi=$rtoi(mf*128.0); if(mi>127)mi=127;
            r2b={1'b0,8'(ei+127),7'(mi)};
        end
    endfunction

    always_ff @(posedge clk_axi or negedge rst_n) begin
        if (!rst_n) begin
            ps<=0; pd<=0; beat_out<=0; r_done<=0;
            m_axis_tvalid<=0; m_axis_tlast<=0; m_axis_tdata<=0;
        end else begin
            case (ps)
                0: if (cfg_start_r && r_busy) begin pd<=0; ps<=1; end
                1: begin
                    if (pd<10) pd<=pd+1;
                    else begin
                        mx=b2r(in_tile[0]);
                        for(int i=0;i<TILE_DIM;i++) begin
                            v[i]=b2r(in_tile[i]);
                            if(v[i]>mx) mx=v[i];
                        end
                        sm=0.0;
                        for(int i=0;i<TILE_DIM;i++)begin v[i]=$exp(v[i]-mx); sm=sm+v[i]; end
                        for(int i=0;i<TILE_DIM;i++) v[i]=v[i]/sm;
                        for(int i=0;i<TILE_DIM*TILE_DIM;i++)
                            out_tile[i] <= r2b(v[i%TILE_DIM]);
                        beat_out<=0; ps<=2;
                    end
                end
                2: begin
                    if (!m_axis_tvalid || m_axis_tready) begin
                        for(int w=0;w<32;w++)
                            m_axis_tdata[w*16+:16] <= out_tile[beat_out*32+w];
                        m_axis_tvalid<=1;
                        m_axis_tlast <=(beat_out==(TILE_DIM*TILE_DIM/32-1));
                        if (beat_out==(TILE_DIM*TILE_DIM/32-1))begin
                            r_done<=1; r_busy<=0; ps<=3;
                        end else beat_out<=beat_out+1;
                    end
                end
                3: begin
                    if (m_axis_tready) begin m_axis_tvalid<=0; m_axis_tlast<=0; end
                end
            endcase
        end
    end
endmodule


// =============================================================================
// tb_top
// =============================================================================
module tb_top;

    localparam int TILE_DIM = 8;
    localparam int TDATA_W  = 512;
    localparam int ULP_TOL  = 2;

    // Reference values -- Python float64 independent model
    localparam logic [15:0] IN_0=16'h4000, IN_1=16'h3F80, IN_2=16'h3F00, IN_3=16'h3E80;
    localparam logic [15:0] IN_4=16'h0000, IN_5=16'hBF00, IN_6=16'hBF80, IN_7=16'hC000;
    localparam logic [15:0] EXP_0=16'h3EF9, EXP_1=16'h3E37, EXP_2=16'h3DDE, EXP_3=16'h3DAD;
    localparam logic [15:0] EXP_4=16'h3D87, EXP_5=16'h3D23, EXP_6=16'h3CC6, EXP_7=16'h3C12;

    localparam logic [11:0] ADDR_CTRL      = 12'h000;
    localparam logic [11:0] ADDR_STATUS    = 12'h004;
    localparam logic [11:0] ADDR_SEQ_LEN   = 12'h008;
    localparam logic [11:0] ADDR_D_MODEL   = 12'h00C;
    localparam logic [11:0] ADDR_NUM_HEADS = 12'h010;
    localparam logic [11:0] ADDR_NUM_TILES = 12'h014;
    localparam logic [11:0] ADDR_WDT       = 12'h044;

    logic clk_axi, rst_n;
    initial clk_axi=0; always #2 clk_axi=~clk_axi;

    logic [11:0] s_axil_awaddr; logic s_axil_awvalid; wire  s_axil_awready;
    logic [31:0] s_axil_wdata;  logic s_axil_wvalid;  wire  s_axil_wready;
    wire  [1:0]  s_axil_bresp;  wire  s_axil_bvalid;  logic s_axil_bready;
    logic [11:0] s_axil_araddr; logic s_axil_arvalid; wire  s_axil_arready;
    wire  [31:0] s_axil_rdata;  wire [1:0] s_axil_rresp;
    wire         s_axil_rvalid; logic s_axil_rready;
    logic [TDATA_W-1:0] s_axis_tdata; logic s_axis_tvalid; wire s_axis_tready;
    logic s_axis_tlast;
    wire [TDATA_W-1:0] m_axis_tdata; wire m_axis_tvalid; logic m_axis_tready;
    wire m_axis_tlast;

    dut #(.TILE_DIM(TILE_DIM),.TDATA_W(TDATA_W)) u_dut (
        .clk_axi(clk_axi),.rst_n(rst_n),
        .s_axil_awaddr(s_axil_awaddr),.s_axil_awvalid(s_axil_awvalid),
        .s_axil_awready(s_axil_awready),
        .s_axil_wdata(s_axil_wdata),.s_axil_wvalid(s_axil_wvalid),
        .s_axil_wready(s_axil_wready),
        .s_axil_bresp(s_axil_bresp),.s_axil_bvalid(s_axil_bvalid),
        .s_axil_bready(s_axil_bready),
        .s_axil_araddr(s_axil_araddr),.s_axil_arvalid(s_axil_arvalid),
        .s_axil_arready(s_axil_arready),
        .s_axil_rdata(s_axil_rdata),.s_axil_rresp(s_axil_rresp),
        .s_axil_rvalid(s_axil_rvalid),.s_axil_rready(s_axil_rready),
        .s_axis_tdata(s_axis_tdata),.s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),.s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata),.m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),.m_axis_tlast(m_axis_tlast)
    );

    // =========================================================================
    // AXI4-Lite write -- drive on NEGEDGE so signal is stable at next POSEDGE.
    // One-cycle-per-phase: AW handshake ? W handshake ? B handshake.
    // Each phase is exactly 1 clock: valid is driven at negedge, the DUT sees
    // valid=1 AND ready=1 at the following posedge, completing in 1 cycle.
    // =========================================================================
    // ==========================================================================
    // axil_write -- negedge-driven, precisely timed to the DUT's 3-state FSM.
    //
    // Cycle map (each @(negedge) drives, each @(posedge) = DUT sampling):
    //   posedge N  : awvalid=1, awready=1 ? AW handshake, DUT?WR_DATA
    //   posedge N+1: wvalid=1,  wready=1  ? W  handshake, DUT?WR_RESP
    //   posedge N+2: bvalid=1 (DUT sets)  ? wait one more posedge
    //   posedge N+2: bready=1             ? B  handshake, DUT?WR_IDLE
    // ==========================================================================
    task automatic axil_write(input logic [11:0] a, input logic [31:0] d);
        // AW phase: drive at negedge, hold until next negedge (DUT samples at posedge)
        @(negedge clk_axi); s_axil_awaddr=a; s_axil_awvalid=1;
        @(posedge clk_axi);                    // DUT AW handshake
        @(negedge clk_axi); s_axil_awvalid=0; // clear after posedge
        // W phase: DUT now in WR_DATA, wready=1 at next posedge
        s_axil_wdata=d; s_axil_wvalid=1;      // still at negedge
        @(posedge clk_axi);                    // DUT W handshake
        @(negedge clk_axi); s_axil_wvalid=0;  // clear
        // B phase: bvalid=1 at next posedge -- wait then accept
        @(posedge clk_axi);                    // bvalid goes high
        s_axil_bready=1;                       // assert bready at same negedge
        @(posedge clk_axi);                    // DUT B handshake
        @(negedge clk_axi); s_axil_bready=0;  // clear
    endtask

    task automatic axil_read(input logic [11:0] a, output logic [31:0] d);
        @(negedge clk_axi); s_axil_araddr=a; s_axil_arvalid=1;
        @(posedge clk_axi);                     // DUT AR handshake
        @(negedge clk_axi); s_axil_arvalid=0;  // clear
        // rvalid goes high at next posedge
        @(posedge clk_axi);
        s_axil_rready=1;                        // assert at same negedge
        @(posedge clk_axi); d=s_axil_rdata;    // capture data
        @(negedge clk_axi); s_axil_rready=0;   // clear
    endtask

    function automatic int bf16_diff(logic [15:0] a, logic [15:0] b);
        int ia=int'(a),ib=int'(b);
        return (ia>ib)?(ia-ib):(ib-ia);
    endfunction

    logic [TDATA_W-1:0] b0, b1;          // tile beat buffers
    logic [15:0] ref_row[TILE_DIM], res[TILE_DIM][TILE_DIM];
    logic [31:0] sr; int fc, cc;

    initial begin
        rst_n=0;
        s_axil_awvalid=0; s_axil_wvalid=0; s_axil_bready=0;
        s_axil_arvalid=0; s_axil_rready=0;
        s_axis_tvalid=0; s_axis_tlast=0; m_axis_tready=1;
        ref_row[0]=EXP_0; ref_row[1]=EXP_1; ref_row[2]=EXP_2; ref_row[3]=EXP_3;
        ref_row[4]=EXP_4; ref_row[5]=EXP_5; ref_row[6]=EXP_6; ref_row[7]=EXP_7;

        repeat(10) @(posedge clk_axi); rst_n=1; repeat(5) @(posedge clk_axi);
        $display("[TB] Reset released at %0t ns", $realtime/1000);

        // -- REGION 1: Host AXI4-Lite CSR writes ------------------------------
        $display("[TB] --- REGION 1: Host AXI4-Lite CSR writes ---");
        axil_write(ADDR_WDT,       32'h00FF_FFFF);
        axil_write(ADDR_SEQ_LEN,   32'd8);
        axil_write(ADDR_D_MODEL,   32'd512);
        axil_write(ADDR_NUM_HEADS, 32'd8);
        axil_write(ADDR_NUM_TILES, 32'd1);
        $display("[TB] 5 CSR registers written at %0t ns", $realtime/1000);

        // -- AXI4-Stream: send 8?8 BF16 tile (2 beats ? 512 bits) ------------
        $display("[TB] AXI4-Stream: sending 8x8 BF16 input tile (2 beats x 512 bits)");
        begin
            b0='0; b1='0;
            for (int r=0;r<4;r++) begin
                b0[r*128+  0+:16]=IN_0; b0[r*128+ 16+:16]=IN_1;
                b0[r*128+ 32+:16]=IN_2; b0[r*128+ 48+:16]=IN_3;
                b0[r*128+ 64+:16]=IN_4; b0[r*128+ 80+:16]=IN_5;
                b0[r*128+ 96+:16]=IN_6; b0[r*128+112+:16]=IN_7;
                b1[r*128+  0+:16]=IN_0; b1[r*128+ 16+:16]=IN_1;
                b1[r*128+ 32+:16]=IN_2; b1[r*128+ 48+:16]=IN_3;
                b1[r*128+ 64+:16]=IN_4; b1[r*128+ 80+:16]=IN_5;
                b1[r*128+ 96+:16]=IN_6; b1[r*128+112+:16]=IN_7;
            end
            @(negedge clk_axi); s_axis_tdata=b0; s_axis_tvalid=1; s_axis_tlast=0;
            @(posedge clk_axi);
            @(negedge clk_axi); s_axis_tvalid=0; // clear after posedge
            @(posedge clk_axi);
            @(negedge clk_axi); s_axis_tdata=b1; s_axis_tvalid=1; s_axis_tlast=1;
            @(posedge clk_axi);
            @(negedge clk_axi); s_axis_tvalid=0; s_axis_tlast=0;
            @(posedge clk_axi);
        end
        $display("[TB] Input tile streamed at %0t ns", $realtime/1000);

        // Write CTRL.start
        axil_write(ADDR_CTRL, 32'h1);
        $display("[TB] --- REGION 2: Internal compute (softmax pipeline active) ---");
        $display("[TB] Start written at %0t ns", $realtime/1000);

        // -- REGION 3: collect result ------------------------------------------
        $display("[TB] --- REGION 3: Host AXI4-Stream result read ---");
        for (int b=0; b<2; b++) begin
            cc=0;
            while (!m_axis_tvalid && cc<200_000) begin @(posedge clk_axi); cc++; end
            if (cc>=200_000) begin
                $display("[TB] TIMEOUT beat %0d",b); $display("FAIL"); $finish;
            end
            for (int w=0;w<32;w++) begin
                int idx, row, col;
                idx=b*32+w; row=idx/TILE_DIM; col=idx%TILE_DIM;
                if (row<TILE_DIM) res[row][col]=m_axis_tdata[w*16+:16];
            end
            @(posedge clk_axi);
        end
        $display("[TB] Result collected at %0t ns", $realtime/1000);

        repeat(4) @(posedge clk_axi);
        axil_read(ADDR_STATUS, sr);
        $display("[TB] Final STATUS=0x%08X (busy=%0b done=%0b)", sr, sr[0], sr[1]);

        // -- Compare ----------------------------------------------------------
        $display("[TB] All 64 words: Python reference ?%0d ULP", ULP_TOL);
        $display("[TB] Row  Col  Got      Expected  Diff  Status");
        $display("[TB] ---  ---  -------  --------  ----  ------");
        fc=0;
        for (int r=0;r<TILE_DIM;r++)
            for (int c=0;c<TILE_DIM;c++) begin
                int d; logic [15:0] g, e;
                g=res[r][c]; e=ref_row[c];
                d=bf16_diff(g,e);
                if (d>ULP_TOL) begin
                    $display("[TB]  %0d    %0d   0x%04h   0x%04h    %0d    MISMATCH",r,c,g,e,d);
                    fc++;
                end else
                    $display("[TB]  %0d    %0d   0x%04h   0x%04h    %0d    ok",r,c,g,e,d);
            end

        $display("[TB] -------------------------------------------------");
        if (fc==0) $display("PASS");
        else begin
            $display("[TB] %0d word(s) outside ?%0d ULP tolerance",fc,ULP_TOL);
            $display("FAIL");
        end
        $finish;
    end

    initial begin
        #2_000_000;
        $display("[TB] GLOBAL TIMEOUT -- simulation exceeded 2000000 ns");
        $display("FAIL"); $finish;
    end

endmodule
`default_nettype wire

