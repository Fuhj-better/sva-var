// test_data_types.sv
module test_data_types (
    input  logic               clk,
    input  logic               rst_n,
    input  logic signed [15:0] signed_data,
    input  logic [15:0]        unsigned_data,
    input  logic [7:0]         byte_array [0:3],
    output logic [31:0]        result_32bit,
    output logic [15:0]        result_16bit,
    output logic               comparison
);

    logic signed [15:0] signed_temp;
    logic [31:0]        multiply_result;
    logic [15:0]        concatenated;

    // 有符号运算
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            signed_temp <= 16'sh0000;
        end else begin
            signed_temp <= signed_data + 16'sh0100;  // 有符号加法
        end
    end

    // 混合符号运算
    assign multiply_result = signed_temp * $signed(unsigned_data);

    // 位选择和部分选择
    assign concatenated = {byte_array[0], byte_array[1]};

    // 复杂表达式
    always_comb begin
        result_32bit = multiply_result;
        result_16bit = concatenated ^ unsigned_data[7:0];  // 位异或
        
        // 复杂比较
        comparison = (signed_data > 16'sh0020) && 
                     (unsigned_data < 16'h0100) &&
                     (|byte_array[2]);  // 归约或
    end

    // 数组操作
    logic [3:0] parity_bits;
    always_comb begin
        for (int i = 0; i < 4; i = i + 1) begin
            parity_bits[i] = ^byte_array[i];  // 每个字节的奇偶校验
        end
    end

endmodule