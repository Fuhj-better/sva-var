// 模拟类型定义 (需要工具能正确解析这些类型和常量)
parameter int WIDTH = 8;
parameter int MODE  = 1; // 1: ENABLE, 0: DISABLE

typedef enum logic [1:0] {
    OP_ADD = 2'h0,
    OP_SUB = 2'h1,
    OP_MUX = 2'h2
} operation_e;

module test_5_param_enum #(
    parameter int DATA_WIDTH = WIDTH 
) (
    input  operation_e op_i,
    input  logic [DATA_WIDTH-1:0] a_i,
    input  logic [DATA_WIDTH-1:0] b_i,
    output logic [DATA_WIDTH-1:0] result_o
);
  // 赋值逻辑依赖于参数和枚举
  always_comb begin
    result_o = '0;
    if (MODE == 1) begin                     // Path 1 (Constant Condition: True)
      if (op_i == OP_ADD) begin
        result_o = a_i + b_i;
      end else if (op_i == OP_SUB) begin
        result_o = a_i - b_i;
      end else begin
        result_o = a_i;
      end
    end else begin                           // Path 2 (Constant Condition: False)
      result_o = b_i;
    end
  end
endmodule