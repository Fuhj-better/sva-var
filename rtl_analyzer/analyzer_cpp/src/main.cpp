#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

#include "slang/ast/Compilation.h"
#include "slang/syntax/SyntaxTree.h"
#include "DataModel.h"
#include "DependencyVisitor.h"
#include "json.hpp"
#include "slang/diagnostics/DiagnosticEngine.h"  
#include "slang/diagnostics/TextDiagnosticClient.h"  

struct TestCase {
        std::string name;
        std::string topModule;
        std::vector<std::string> sourceFiles;
        std::vector<std::string> headerFiles;
        std::string outputPath;
    };

std::vector<TestCase> testSuite = {
        // --- 现有测试用例 ---
        // {
        //     "Basic Dataflow",
        //     "test_basic",
        //     {"../test_suite/1_basic_dataflow/test_basic.sv"},
        //     {},
        //     "../../results/1_basic_dataflow.json"
        // },
        // {
        //     "Sequential Controlflow",
        //     "test_sequential",
        //     {"../../test_suite/2_sequential_controlflow/test_sequential.sv"},
        //     {},
        //     "../../results/2_sequential_controlflow.json"
        // },
        // {
        //     "Module Hierarchy",
        //     "top_module",
        //     {
        //         "../../test_suite/3_module_hierarchy/sub_module.sv",
        //         "../../test_suite/3_module_hierarchy/top_module.sv"
        //     },
        //     {},
        //     "../../results/3_module_hierarchy.json"
        // },
        // {
        //     "Complex Conditions",
        //     "test_complex_conditions",
        //     {"../test_suite/4_complex_conditions/test_complex_conditions.sv"},
        //     {},
        //     "../../results/4_complex_conditions.json"
        // },
        // {
        //     "Complex Hierarchy",
        //     "top_hierarchy",
        //     {"../../test_suite/5_complex_hierarchy/hierarchy.sv"},
        //     {},
        //     "../../results/5_complex_hierarchy.json"
        // },
        // {
        //     "Control Flow",
        //     "test_control_flow",
        //     {"../test_suite/6_controlflow/test_control_flow.sv"},
        //     {},
        //     "../../results/6_controlflow.json"
        // },
        // {
        //     "Data Types",
        //     "test_data_types",
        //     {"../test_suite/7_datatypes/test_data_types.sv"},
        //     {},
        //     "../../results/7_datatypes.json"
        // }
        
        // --- 新增 Ibex 测试用例 ---
        // {
        //     "Ibex Compressed Decoder",
        //     "ibex_compressed_decoder",
        //     {"/data/fhj/sva-var/ibex/rtl/ibex_compressed_decoder.sv"},
        //     {
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert_dummy_macros.svh",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert_yosys_macros.svh",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert_standard_macros.svh",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert_sec_cm.svh",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_flop_macros.sv",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert.sv",
        //         "/data/fhj/sva-var/ibex/rtl/ibex_pkg.sv"

        //     },
        //     "../../results/ibex_compressed_decoder.json"
        // },
        // {
        //     "Ibex ALU",
        //     "ibex_alu",
        //     {"/data/fhj/sva-var/ibex/rtl/ibex_alu.sv"},
        //     {
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert_dummy_macros.svh",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert_yosys_macros.svh",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert_standard_macros.svh",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert_sec_cm.svh",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_flop_macros.sv",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert.sv",
        //         "/data/fhj/sva-var/ibex/rtl/ibex_pkg.sv"

        //     },
        //     "../../results/ibex_alu.json"
        // },
        // {
        //     "Ibex Core",
        //     "ibex_alu", // 假设顶层模块名是 ibex_core
        //     {
        //         "/data/fhj/sva-var/ibex/rtl/ibex_alu.sv"
        //     },
        //     {   
        //         "/data/fhj/sva-var/ibex/rtl/ibex_pkg.sv",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert_dummy_macros.svh",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert_yosys_macros.svh",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert_standard_macros.svh",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert_sec_cm.svh",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_flop_macros.sv",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert.sv",
        //         "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/dv/sv/dv_utils/dv_fcov_macros.svh"
                
        //     },
        //     "../../results/ibex_alu.json"
        // },
        {
            "Ibex Core",
            "ibex_core", // 假设顶层模块名是 ibex_core
            {
                "/data/fhj/sva-var/ibex/rtl/ibex_alu.sv",
                "/data/fhj/sva-var/ibex/rtl/ibex_compressed_decoder.sv",
                "/data/fhj/sva-var/ibex/rtl/ibex_controller.sv",
                "/data/fhj/sva-var/ibex/rtl/ibex_counter.sv",
                "/data/fhj/sva-var/ibex/rtl/ibex_cs_registers.sv",
                "/data/fhj/sva-var/ibex/rtl/ibex_decoder.sv",
                "/data/fhj/sva-var/ibex/rtl/ibex_ex_block.sv",
                "/data/fhj/sva-var/ibex/rtl/ibex_id_stage.sv",
                "/data/fhj/sva-var/ibex/rtl/ibex_if_stage.sv",
                "/data/fhj/sva-var/ibex/rtl/ibex_load_store_unit.sv",
                "/data/fhj/sva-var/ibex/rtl/ibex_multdiv_slow.sv",
                "/data/fhj/sva-var/ibex/rtl/ibex_multdiv_fast.sv",
                "/data/fhj/sva-var/ibex/rtl/ibex_prefetch_buffer.sv",
                "/data/fhj/sva-var/ibex/rtl/ibex_fetch_fifo.sv",
                "/data/fhj/sva-var/ibex/rtl/ibex_register_file_ff.sv",
                "/data/fhj/sva-var/ibex/rtl/ibex_csr.sv", 
                "/data/fhj/sva-var/ibex/rtl/ibex_wb_stage.sv",  
                "/data/fhj/sva-var/ibex/rtl/ibex_core.sv"
            },
            {   
                "/data/fhj/sva-var/ibex/rtl/ibex_pkg.sv",
                "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert_dummy_macros.svh",
                "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert_yosys_macros.svh",
                "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert_standard_macros.svh",
                "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert_sec_cm.svh",
                "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_flop_macros.sv",
                "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_assert.sv",
                "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/dv/sv/dv_utils/dv_fcov_macros.svh"
                
            },
            "../../results/ibex_core.json"
        }
        
        // {
        //     "T1. Basic Dataflow",
        //     "test_1_basic_flow",
        //     {"../../test_suite/test_1_basic_flow.sv"},
        //     {},
        //     "../../results/T1_basic_flow.json"
        // },
        // {
        //     "T2. If-Else Control Path",
        //     "test_2_if_else",
        //     {"../../test_suite/test_2_if_else_path.sv"},
        //     {},
        //     "../../results/T2_if_else_path.json"
        // },
        // {
        //     "T3. Case Statement & Nested If",
        //     "test_3_case",
        //     {"../../test_suite/test_3_case_statement.sv"},
        //     {},
        //     "../../results/T3_case_statement.json"
        // },
        // {
        //     "T4. Generate Loop & Genvar",
        //     "test_4_generate",
        //     {"../../test_suite/test_4_generate_loop.sv"},
        //     {},
        //     "../../results/T4_generate_loop.json"
        // },
        // {
        //     "T5. Parameter & Enum Logic",
        //     "test_5_param_enum",
        //     {"../../test_suite/test_5_param_enum.sv"},
        //     {},
        //     "../../results/T5_param_enum.json"
        // },
        // {
        //     "T6. Module Hierarchy & Port Connection",
        //     "test_6_module_inst",
        //     {"../../test_suite/test_6_module_inst.sv"},
        //     {},
        //     "../../results/T6_module_inst.json"
        // },
        // {
        //     "T7. Sequential Logic & Complex Path",
        //     "test_7_multicycle",
        //     {"../../test_suite/test_7_multicycle.sv"},
        //     {},
        //     "../../results/T7_multicycle.json"
        // },
        // {
        //     "T8. Array & Selects",
        //     "test_8_array_select",
        //     {"../../test_suite/test_8_array_select.sv"},
        //     {},
        //     "../../results/T8_array_select.json"
        // },
        // {
        //     "T9. Struct & Union Access",
        //     "test_9_struct_union",
        //     {"../../test_suite/test_9_struct_union.sv"},
        //     {},
        //     "../../results/T9_struct_union.json"
        // },
        // {
        //     "T10. Interface & Modport",
        //     "test_10_interface",
        //     {"../../test_suite/test_10_interface.sv"},
        //     // {"../test_suite/simple_bus_if.sv"}, // 接口文件作为头文件
        //     {},
        //     "../../results/T10_interface.json"
        // }
    };


using json = nlohmann::json;

void to_json(json& j, const ConditionExpression& expr) {
    j = json{
        {"expression", expr.expression},
        {"involvedSignals", expr.involvedSignals},
        {"involvedParameters", expr.involvedParameters} // ←← 新增
    };
}

void to_json(json& j, const ConditionClause& c) {
    j = json{
        {"expr", c.expr},
        {"polarity", c.polarity}
    };
}

void to_json(json& j, const AssignmentInfo& a) {
    j = json{
        {"path", a.path},
        {"drivingSignals", a.drivingSignals},
        {"file", a.file},
        {"line", a.line},
        {"type", a.type},
        {"logicType", a.logicType},
        {"conditionDepth", a.conditionDepth}
    };
}

void to_json(json& j, const VariableInfo& v) {
    j = json{
        {"fullName", v.fullName},
        {"type", v.type},
        {"file", v.fileName},
        {"line", v.line},
        {"direction", v.direction},
        {"bitWidth", v.bitWidth},
        {"assignments", v.assignments},
        {"fanOut", v.fanOut},
        {"assignmentCount", v.assignmentCount},
        {"drivesOutput", v.drivesOutput},
        {"isControlVariable", v.isControlVariable}
    };
}

// --- 核心分析函数 ---
bool runAnalysis(const std::string& topModule,   
                 const std::vector<std::string>& sourceFiles,  
                 const std::vector<std::string>& headerFiles,  
                 const std::string& outputPath) {  
      
    std::cout << "--- Analyzing Test Case: " << topModule << " ---" << std::endl;  
    std::cout << "Source files: " << sourceFiles.size() << std::endl;  
    std::cout << "Header files: " << headerFiles.size() << std::endl;  
  
    std::filesystem::path outPath(outputPath);  
    if (outPath.has_parent_path()) {  
        std::filesystem::create_directories(outPath.parent_path());  
    }  
  
    // 创建 SourceManager  
    slang::SourceManager sourceManager;  
      
    // 配置预处理器选项,添加 include 路径  
    slang::parsing::PreprocessorOptions ppOptions; 
     
    ppOptions.additionalIncludePaths = {  
        "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/ip/prim/rtl",  
        "/data/fhj/sva-var/ibex/rtl",
        "/data/fhj/sva-var/ibex/vendor/lowrisc_ip/dv/sv/dv_utils" 
    };  
      
    slang::Bag options;  
    options.set(ppOptions);  
      
    // 重要:使用带 options 的 Compilation 构造函数  
    slang::ast::Compilation compilation(options);  
      
    // 首先添加头文件 - 传递 sourceManager 和 options  
    for (const auto& headerFile : headerFiles) {  
        if (!std::filesystem::exists(headerFile)) {  
            std::cerr << "[WARNING] Header file not found: " << headerFile << std::endl;  
            continue;  
        }  
        // 关键修改:传递 sourceManager 和 options  
        auto tree_expected = slang::syntax::SyntaxTree::fromFile(headerFile, sourceManager, options);  
        if (!tree_expected) {  
            std::cerr << "[WARNING] Failed to parse header file: " << headerFile << std::endl;  
            continue;  
        }  
        compilation.addSyntaxTree(*tree_expected);  
        std::cout << "[INFO] Added header file: " << headerFile << std::endl;  
    }  
      
    // 然后添加源文件 - 同样传递 sourceManager 和 options  
    for (const auto& sourceFile : sourceFiles) {  
        if (!std::filesystem::exists(sourceFile)) {  
            std::cerr << "[ERROR] Source file not found: " << sourceFile << std::endl;  
            return false;  
        }  
        // 关键修改:传递 sourceManager 和 options  
        auto tree_expected = slang::syntax::SyntaxTree::fromFile(sourceFile, sourceManager, options);  
        if (!tree_expected) {  
            std::cerr << "[ERROR] Failed to parse source file: " << sourceFile << std::endl;  
            return false;  
        }  
        compilation.addSyntaxTree(*tree_expected);  
        std::cout << "[INFO] Added source file: " << sourceFile << std::endl;  
    }  
      
    auto& root = compilation.getRoot();  
      
    // 输出顶层模块  
    auto topInstances = root.topInstances;  
    if (!topInstances.empty()) {  
        std::cout << "Top level design units:\n";  
        for (auto inst : topInstances)  
            std::cout << "    " << inst->name << "\n";  
        std::cout << "\n";  
    }  
      
    // 创建诊断引擎和客户端  
    slang::DiagnosticEngine diagEngine(sourceManager);  
    auto textClient = std::make_shared<slang::TextDiagnosticClient>();  
    diagEngine.addClient(textClient);  
      
    // 发出所有诊断  
    for (auto& diag : compilation.getAllDiagnostics())  
        diagEngine.issue(diag);  
      
    // 输出诊断结果  
    std::cout << textClient->getString();  
      
    // 检查是否有错误  
    int numErrors = diagEngine.getNumErrors();  
    if (numErrors > 0) {  
        std::cerr << "[FAILED] Build failed: " << numErrors << " errors\n";  
        return false;  
    }  
      
    std::cout << "[SUCCESS] Build succeeded\n";  
      
    // 查找顶层模块  
    const slang::ast::InstanceSymbol* top = nullptr;  
    for (auto inst : root.topInstances) {  
        if (inst->name == topModule) {  
            top = inst;  
            break;  
        }  
    }  
      
    if (!top) {  
        std::cerr << "[ERROR] Top module '" << topModule << "' not found." << std::endl;  
        return false;  
    }  
      
    DependencyVisitor visitor;  
    top->visit(visitor);  
    visitor.postProcess();  
      
    json resultJson = visitor.getResults();  
  
    std::ofstream outFile(outputPath);  
    if (outFile.is_open()) {  
        outFile << resultJson.dump(4);   
        outFile.close();  
        std::cout << "[SUCCESS] Results written to: " << outputPath << std::endl;  
        return true;  
    } else {  
        std::cerr << "[ERROR] Could not open output file for writing: " << outputPath << std::endl;  
        return false;  
    }  
}

// 主函数，作为测试驱动器
int main(int argc, char** argv) {
    

    int successCount = 0;
    for (const auto& testCase : testSuite) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Running: " << testCase.name << std::endl;
        std::cout << "========================================" << std::endl;
        
        if (runAnalysis(testCase.topModule, testCase.sourceFiles, testCase.headerFiles, testCase.outputPath)) {
            successCount++;
            std::cout << "✓ PASSED: " << testCase.name << std::endl;
        } else {
            std::cerr << "✗ FAILED: " << testCase.name << std::endl;
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Suite Finished." << std::endl;
    std::cout << "Passed: " << successCount << " / " << testSuite.size() << std::endl;
    std::cout << "========================================" << std::endl;

    return (successCount == testSuite.size()) ? 0 : 1;
}