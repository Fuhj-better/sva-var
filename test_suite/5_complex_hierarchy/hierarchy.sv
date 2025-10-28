// sub_module_a.sv
module sub_module_a (
    input  logic [7:0]  data_in,
    input  logic        enable,
    input  logic        mode_sel,
    output logic [7:0]  data_out,
    output logic        status
);

    logic [7:0] processed_data;

    always_comb begin
        if (enable) begin
            if (mode_sel)
                processed_data = data_in << 1;
            else
                processed_data = data_in >> 1;
        end else begin
            processed_data = 8'h00;
        end
    end

    assign data_out = processed_data;
    assign status = |data_in;  // 按位或归约

endmodule

// sub_module_b.sv  
module sub_module_b (
    input  logic [7:0]  a,
    input  logic [7:0]  b,
    input  logic [1:0]  op_sel,
    output logic [7:0]  result,
    output logic        overflow
);

    always_comb begin
        case (op_sel)
            2'b00: result = a + b;
            2'b01: result = a - b;
            2'b10: result = a & b;
            2'b11: result = a | b;
            default: result = 8'h00;
        endcase
    end

    assign overflow = (op_sel == 2'b00) ? (result < a) : 1'b0;

endmodule

// top_hierarchy.sv
module top_hierarchy (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [7:0]  main_data,
    input  logic [1:0]  control,
    input  logic [7:0]  aux_data,
    output logic [7:0]  final_result,
    output logic [1:0]  status_flags
);

    logic [7:0] stage1_out;
    logic       stage1_status;
    logic [7:0] stage2_out;
    logic       stage2_overflow;

    // 第一级处理
    sub_module_a u_sub_a (
        .data_in(main_data),
        .enable(~control[0]),
        .mode_sel(control[1]),
        .data_out(stage1_out),
        .status(stage1_status)
    );

    // 第二级处理  
    sub_module_b u_sub_b (
        .a(stage1_out),
        .b(aux_data),
        .op_sel(control),
        .result(stage2_out),
        .overflow(stage2_overflow)
    );

    // 输出寄存器
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            final_result <= 8'h00;
            status_flags <= 2'b00;
        end else begin
            final_result <= stage2_out;
            status_flags <= {stage2_overflow, stage1_status};
        end
    end

endmodule