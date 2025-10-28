module sub_module (
    input  logic [7:0] data_in,
    output logic [7:0] data_out
);
  logic [7:0] temp_sig;
  assign temp_sig = data_in;               // data_in -> temp_sig (Internal)
  assign data_out = temp_sig ^ 8'hF0;      // temp_sig -> data_out (Internal)
endmodule

module test_6_module_inst (
    input  logic [7:0] ext_data_i,
    input  logic [7:0] ext_ctrl_i,
    output logic [7:0] ext_result_o
);
  logic [7:0] mid_wire;

  // 实例化: 检查端口连接的依赖
  sub_module u_sub (
    .data_in (ext_data_i),                 // ext_data_i -> u_sub.data_in (Port Connection)
    .data_out(mid_wire)                    // u_sub.data_out -> mid_wire (Port Connection)
  );

  // 最终赋值
  assign ext_result_o = mid_wire & ext_ctrl_i; // mid_wire, ext_ctrl_i -> ext_result_o
endmodule