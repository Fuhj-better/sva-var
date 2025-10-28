module test_8_array_select (
    input  logic [7:0] data_bundle_i [3:0], // 数组输入
    input  logic [1:0] select_i,            
    input  logic [7:0] mask_i,              
    output logic [7:0] element_o
);
  logic [7:0] read_data;

  // 1. 数组元素选择 (Element Select)
  // 依赖: data_bundle_i (整体), select_i -> read_data
  assign read_data = data_bundle_i[select_i]; 

  // 2. 位选择 (Range Select)
  logic [3:0] low_nibble;
  assign low_nibble = read_data[3:0]; 

  // 3. 数组作为整体被驱动
  logic [7:0] output_array [3:0];
  assign output_array = data_bundle_i; 

  // 4. 最终结果
  always_comb begin
    element_o = read_data & mask_i; 
  end
endmodule