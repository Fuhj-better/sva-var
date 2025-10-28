// 1. 结构体定义
typedef struct packed {
    logic [7:0] addr;
    logic [7:0] data;
    logic       valid;
} packet_s;

// 2. 联合体定义
typedef union {
    logic [15:0] word;
    packet_s     packet;
} bus_data_u;

module test_9_struct_union (
    input  logic [15:0] in_word,
    input  logic [7:0]  in_addr,
    output logic [7:0]  out_data,
    output logic [7:0]  out_addr
);
  packet_s rx_packet;
  bus_data_u bus_data;

  // --- 赋值 1: 结构体成员赋值 (直接) ---
  assign rx_packet.addr = in_addr;

  // --- 赋值 2: 联合体成员赋值 (别名) ---
  assign bus_data.word = in_word;

  // --- 赋值 3: 联合体成员访问 (多级) ---
  // 依赖: bus_data (整体) -> out_data (通过 .packet.data 别名)
  assign out_data = bus_data.packet.data; 

  // --- 赋值 4: 结构体成员访问 (直接) ---
  // 依赖: rx_packet.addr -> out_addr
  assign out_addr = rx_packet.addr;
endmodule