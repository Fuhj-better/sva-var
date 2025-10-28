// test_control_flow.sv
module test_control_flow (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [2:0]  cmd,
    input  logic [15:0] data_in,
    input  logic        valid_in,
    output logic [15:0] data_out,
    output logic        ready_out,
    output logic        error_out
);

    typedef enum logic [2:0] {
        IDLE    = 3'b000,
        READ    = 3'b001, 
        PROCESS = 3'b010,
        WRITE   = 3'b011,
        ERROR   = 3'b100
    } state_t;

    state_t current_state, next_state;
    logic [15:0] buffer_reg;
    logic [3:0]  counter;
    logic        timeout;

    // 状态机
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
            buffer_reg <= 16'h0000;
            counter <= 4'b0000;
        end else begin
            current_state <= next_state;
            
            case (current_state)
                READ: begin
                    if (valid_in)
                        buffer_reg <= data_in;
                end
                PROCESS: begin
                    buffer_reg <= buffer_reg + 16'h0001;
                    counter <= counter + 1'b1;
                end
                default: begin
                    // 保持原值
                end
            endcase
        end
    end

    // 下一状态逻辑
    always_comb begin
        next_state = current_state;
        case (current_state)
            IDLE: begin
                if (cmd == 3'b001 && valid_in)
                    next_state = READ;
                else if (cmd == 3'b010)
                    next_state = PROCESS;
            end
            READ: begin
                if (valid_in)
                    next_state = PROCESS;
                else if (timeout)
                    next_state = ERROR;
            end
            PROCESS: begin
                if (counter == 4'b1111)
                    next_state = WRITE;
                else if (buffer_reg[15])
                    next_state = ERROR;
            end
            WRITE: next_state = IDLE;
            ERROR: next_state = IDLE;
        endcase
    end

    // 输出逻辑
    assign data_out = (current_state == WRITE) ? buffer_reg : 16'h0000;
    assign ready_out = (current_state == IDLE) || (current_state == WRITE);
    assign error_out = (current_state == ERROR);
    assign timeout = (counter > 4'b1000);

endmodule