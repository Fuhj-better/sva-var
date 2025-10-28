module test_1_basic_flow (
    input  logic [7:0] A_in,
    input  logic [7:0] B_in,
    output logic [7:0] Z_out
);
  logic [7:0] data_reg;
  logic [7:0] combinational_sig;

  // 连续赋值 (Assign Statement)
  assign combinational_sig = A_in + B_in; 

  // 组合逻辑 (Always_comb)
  always_comb begin
    data_reg = combinational_sig;         
  end

  // 另一个组合逻辑块
  always_comb begin
    Z_out = data_reg;                    
  end

endmodule