interface simple_bus_if;
    logic [7:0] addr;
    logic [7:0] data;
    logic       req;
    
    modport master (output addr, output data, output req);
    modport slave  (input addr, input data, input req);
endinterface


module master_agent (
    input  logic clk,
    simple_bus_if.master bus_if // 接口和 modport 作为端口
);
    logic [7:0] master_data;

    // --- 赋值 1: 接口信号赋值 (内部信号驱动接口信号) ---
    // 依赖: master_data -> bus_if.data
    assign bus_if.data = master_data;

    // --- 赋值 2: 接口信号驱动 (clk 驱动 req) ---
    // 依赖: clk -> bus_if.req
    assign bus_if.req = clk; 

    initial master_data = 8'hAA; 
endmodule


module slave_monitor (
    input  logic clk,
    simple_bus_if.slave bus_if // 接口和 modport 作为端口
);
    // --- 赋值 3: 接口信号读取（Fan-Out） ---
    // 依赖: bus_if.addr -> slave_addr
    logic [7:0] slave_addr;
    assign slave_addr = bus_if.addr; 
endmodule


module test_10_interface (
    input  logic clk_i,
    input  logic reset_i,
    output logic [7:0] monitor_addr_o,
    output logic [7:0] monitor_data_o
);
    // 实例化接口
    simple_bus_if u_bus_if();

    // 实例化 master agent，连接到接口
    master_agent u_master (
        .clk    (clk_i),
        .bus_if (u_bus_if) // 接口连接
    );

    // 实例化 slave monitor，连接到同一接口
    slave_monitor u_slave (
        .clk    (clk_i),
        .bus_if (u_bus_if) 
    );
    
    // --- 赋值 4: 从接口信号到顶层输出的连接 (跨层) ---
    // 依赖: u_bus_if.data -> monitor_data_o
    assign monitor_data_o = u_bus_if.data;

    // 依赖: u_bus_if.addr -> monitor_addr_o
    assign monitor_addr_o = u_bus_if.addr;
endmodule