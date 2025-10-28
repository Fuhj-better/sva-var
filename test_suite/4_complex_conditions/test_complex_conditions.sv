// test_complex_conditions.sv
module test_complex_conditions (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [1:0]  mode,
    input  logic [3:0]  config_bits,
    input  logic [7:0]  data_in_a,
    input  logic [7:0]  data_in_b,
    input  logic [7:0]  data_in_c,
    output logic [7:0]  data_out,
    output logic        valid_out,
    output logic        error_flag
);

    logic [7:0] internal_reg;
    logic [2:0] state;
    logic       enable_processing;

    // 复杂条件赋值
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= 8'h00;
            valid_out <= 1'b0;
            error_flag <= 1'b0;
            state <= 3'b000;
        end else begin
            case (state)
                3'b000: begin
                    if (mode == 2'b00 && config_bits[0]) begin
                        data_out <= data_in_a;
                        state <= 3'b001;
                    end else if (mode == 2'b01 && |config_bits[3:2]) begin
                        data_out <= data_in_b;
                        state <= 3'b010;
                    end else begin
                        data_out <= data_in_c;
                        state <= 3'b011;
                    end
                    valid_out <= 1'b1;
                end
                3'b001: begin
                    if (data_in_a > 8'h80)
                        error_flag <= 1'b1;
                    state <= 3'b000;
                end
                default: state <= 3'b000;
            endcase
        end
    end

    // 组合逻辑与条件
    assign enable_processing = (mode != 2'b11) && (config_bits[1:0] != 2'b00);
    
    // 嵌套条件
    always_comb begin
        if (enable_processing) begin
            if (config_bits[2]) begin
                internal_reg = data_in_a + data_in_b;
            end else begin
                internal_reg = data_in_a - data_in_b;
            end
        end else begin
            internal_reg = 8'hFF;
        end
    end

endmodule