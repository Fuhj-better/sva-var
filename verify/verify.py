import argparse
from collections import OrderedDict
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import copy

from datetime import datetime
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

import re
import json
from pathlib import Path
from typing import Any, Dict, List

def parse_sva_from_json_files_structured(
    sva_files: List[Path], default_module_name: str = "ibex_if_stage"
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    从多个JSON格式的SVA文件中解析所有断言，支持大模型回复的非标准格式。
    处理被 ```json ``` 包围的JSON内容。

    Args:
        sva_files (List[Path]): SVA文件的路径列表。
        default_module_name (str): 目标模块名称，默认为 "ibex_if_stage"。

    Returns:
        Dict[str, Dict[str, List[Dict[str, Any]]]]:
        一个嵌套字典，结构为: {module_name: {variable_name: [sva_dicts]}}
        每个 sva_dict 包含 'sva_string', 'status', 和 'sva_id'。
    """
    structured_svas = {}

    if default_module_name not in structured_svas:
        structured_svas[default_module_name] = {}

    for sva_file in sva_files:
        variable_name = sva_file.stem  # e.g., "boot_addr_i"
        
        if variable_name not in structured_svas[default_module_name]:
            structured_svas[default_module_name][variable_name] = []
        
        try:
            # 读取文件内容
            with open(sva_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 处理大模型回复的非标准格式
            json_content = extract_json_from_llm_response(content)
            
            if not json_content:
                logging.warning(f"No valid JSON content found in {sva_file}")
                continue
                
            # 解析JSON
            data = json.loads(json_content)
            
            # 处理不同的JSON结构
            if isinstance(data, list):
                # 如果是列表格式，直接处理
                sva_list = data
            elif isinstance(data, dict) and "assertions" in data:
                # 如果是字典格式，包含assertions字段
                sva_list = data["assertions"]
            elif isinstance(data, dict) and "svas" in data:
                # 如果是字典格式，包含svas字段
                sva_list = data["svas"]
            else:
                # 其他格式，尝试作为列表处理
                sva_list = [data] if isinstance(data, dict) else []
            
            # 处理每条断言
            for i, item in enumerate(sva_list):
                sva_string = extract_sva_string(item)
                if sva_string:
                    # 创建唯一的、可预测的ID
                    sva_id = f"{default_module_name}_{variable_name}_{i}"
                    
                    sva_dict = {
                        "sva_id": sva_id,
                        "sva_string": sva_string,
                        "status": "unknown",
                        "variable_name": variable_name,
                        "index": i
                    }
                    structured_svas[default_module_name][variable_name].append(sva_dict)
                    
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logging.error(f"Failed to parse {sva_file}: {e}")
            logging.debug(f"Problematic content: {content[:200]}...")
            
    return structured_svas

def extract_json_from_llm_response(content: str) -> str:
    """
    从大模型回复中提取JSON内容。
    处理以下格式：
    1. ```json ... ```
    2. ``` ... ``` (无语言标记)
    3. 纯JSON
    4. 包含解释文本的回复
    """
    # 移除前后的空白字符
    content = content.strip()
    
    # 情况1: 被 ```json ``` 包围
    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    # 情况2: 被 ``` ``` 包围 (无语言标记)
    code_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # 情况3: 尝试直接查找JSON数组或对象
    # 查找以 [ 开头 ] 结尾的内容
    array_match = re.search(r'(\[.*\])', content, re.DOTALL)
    if array_match:
        return array_match.group(1).strip()
    
    # 查找以 { 开头 } 结尾的内容
    object_match = re.search(r'(\{.*\})', content, re.DOTALL)
    if object_match:
        return object_match.group(1).strip()
    
    # 情况4: 如果以上都不匹配，返回原始内容（可能是纯JSON）
    return content

def extract_sva_string(item: Any) -> str:
    """
    从JSON项中提取SVA字符串。
    支持多种可能的字段名。
    """
    if isinstance(item, str):
        return item.strip()
    
    if isinstance(item, dict):
        # 尝试不同的字段名
        possible_fields = [
            "assertion", "sva", "property", "assert", 
            "sva_string", "property_string", "assertion_string"
        ]
        
        for field in possible_fields:
            if field in item and isinstance(item[field], str):
                return item[field].strip()
        
        # 如果没有找到标准字段，尝试第一个字符串值
        for value in item.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
    
    return ""

def insert_single_sva_into_module(
    original_rtl_file: Path,
    target_module_name: str,
    sva_to_insert: str,
    output_rtl_file: Path,
    indent: str = "  ",
) -> bool:
    """
    在Verilog文件的指定模块的endmodule之前插入单个SVA。
    专门用于 ibex_if_stage.sv 文件。

    Args:
        original_rtl_file (Path): 原始RTL文件的路径。
        target_module_name (str): 目标模块的名称。
        sva_to_insert (str): 要插入的SystemVerilog断言字符串。
        output_rtl_file (Path): 输出文件的路径。
        indent (str): 插入SVA时使用的缩进字符串。

    Returns:
        bool: 如果成功插入SVA则返回True，否则返回False。
    """
    try:
        with open(original_rtl_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 查找模块的结束位置
        module_pattern = re.compile(
            rf"^\s*module\s+{re.escape(target_module_name)}\b.*?^endmodule",
            re.MULTILINE | re.DOTALL
        )
        
        match = module_pattern.search(content)
        if not match:
            logging.error(f"Module '{target_module_name}' not found in {original_rtl_file}")
            return False

        module_content = match.group(0)
        
        # 在 endmodule 前插入 SVA
        sva_indented = '\n'.join([indent + line for line in sva_to_insert.strip().split('\n')])
        modified_module = module_content.replace(
            "endmodule",
            f"{sva_indented}\nendmodule"
        )
        
        # 替换整个模块内容
        modified_content = content[:match.start()] + modified_module + content[match.end():]
        
        # 写入输出文件
        with open(output_rtl_file, "w", encoding="utf-8") as f:
            f.write(modified_content)

        logging.info(f"Successfully inserted SVA into {output_rtl_file}")
        return True

    except Exception as e:
        logging.error(f"Error inserting SVA into {original_rtl_file}: {e}")
        return False

def create_ibex_file_list(ibex_rtl_dir: Path) -> List[Path]:
    """
    创建 Ibex 设计所需的文件列表。
    """
    file_list = [
        ibex_rtl_dir / "ibex_pkg.sv",
        ibex_rtl_dir / "vendor/lowrisc_ip/ip/prim/rtl/prim_assert_dummy_macros.svh",
        ibex_rtl_dir / "vendor/lowrisc_ip/ip/prim/rtl/prim_assert_yosys_macros.svh",
        ibex_rtl_dir / "vendor/lowrisc_ip/ip/prim/rtl/prim_assert_standard_macros.svh",
        ibex_rtl_dir / "vendor/lowrisc_ip/ip/prim/rtl/prim_assert_sec_cm.svh",
        ibex_rtl_dir / "vendor/lowrisc_ip/ip/prim/rtl/prim_flop_macros.sv",
        ibex_rtl_dir / "vendor/lowrisc_ip/ip/prim/rtl/prim_assert.sv",
        ibex_rtl_dir / "vendor/lowrisc_ip/dv/sv/dv_utils/dv_fcov_macros.svh",
        ibex_rtl_dir / "ibex_alu.sv",
        ibex_rtl_dir / "ibex_compressed_decoder.sv",
        ibex_rtl_dir / "ibex_controller.sv",
        ibex_rtl_dir / "ibex_counter.sv",
        ibex_rtl_dir / "ibex_cs_registers.sv",
        ibex_rtl_dir / "ibex_decoder.sv",
        ibex_rtl_dir / "ibex_ex_block.sv",
        ibex_rtl_dir / "ibex_id_stage.sv",
        ibex_rtl_dir / "ibex_if_stage.sv",  # 主要目标文件
        ibex_rtl_dir / "ibex_prefetch_buffer.sv",
        ibex_rtl_dir / "ibex_fetch_fifo.sv",
        ibex_rtl_dir / "ibex_register_file_ff.sv",
        ibex_rtl_dir / "ibex_core.sv",
    ]
    
    # 检查文件是否存在
    existing_files = [f for f in file_list if f.exists()]
    missing_files = [f for f in file_list if not f.exists()]
    
    if missing_files:
        logging.warning(f"Missing files: {missing_files}")
    
    return existing_files

def create_fpv_tcl_template():
    """
    创建 FPV TCL 模板。
    """
    return """# FPV TCL script for Ibex if_stage verification

# Set engine parameter to ignore design assertions

# Analyze all required files
analyze -sv12 \\
    +incdir+{prim_assert_dir} \\
    +incdir+{dv_utils_dir} \\
    +incdir+{rtl_dir} \\
    -f {file_list_path}

# Elaborate the top module
elaborate -top {top_module}

# Set clock and reset
clock {clock_signal}
reset {reset_signal}

# Prove all properties
prove -all

# Exit
exit -force
"""

def generate_fpv_tcl(
    output_path: Path,
    prim_assert_dir: Path,
    dv_utils_dir: Path,
    rtl_dir: Path,
    file_list_path: Path,
    top_module: str = "ibex_if_stage",
    clock_signal: str = "clk_i",
    reset_signal: str = "~rst_ni"
) -> bool:
    """
    生成 FPV TCL 脚本。
    """
    try:
        template = create_fpv_tcl_template()
        tcl_content = template.format(
            prim_assert_dir=prim_assert_dir,
            dv_utils_dir=dv_utils_dir,
            rtl_dir=rtl_dir,
            file_list_path=file_list_path,
            top_module=top_module,
            clock_signal=clock_signal,
            reset_signal=reset_signal
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(tcl_content)
        
        logging.info(f"Generated FPV TCL script: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error generating FPV TCL script: {e}")
        return False

def extract_proof_status(report_content: str) -> str:
    """
    从 JasperGold 报告中提取证明状态。
    """
    if "ERROR" in report_content:
        return "error"
    
    summary_section = re.search(
        r"SUMMARY\n==============================================================\n(.*?)\n={2,}",
        report_content,
        re.DOTALL,
    )
    if not summary_section:
        return "inconclusive"

    summary_lines = summary_section.group(1).strip().split("\n")
    
    for line in summary_lines:
        if line.strip().startswith("- proven"):
            return "proven"
        elif line.strip().startswith("- cex"):
            return "cex"
        elif line.strip().startswith("- ar_cex"):
            return "cex"
        elif line.strip().startswith("- undetermined"):
            return "inconclusive"
        elif line.strip().startswith("- unknown"):
            return "inconclusive"
        elif line.strip().startswith("- timeout"):
            return "timeout"
    
    return "inconclusive"

def run_jg_single(tcl_file_path: Path, jg_dir: Path, rpt_dir: Path) -> Path:
    """
    运行单个 JasperGold 验证。
    """
    sva_name = tcl_file_path.stem
    project_dir = jg_dir / sva_name
    
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    jasper_command = f"jg -batch -proj {project_dir} -tcl {tcl_file_path}"
    logging.info(f"Running: {jasper_command}")
    
    process = None
    report = ""

    try:
        process = subprocess.Popen(
            jasper_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        
        stdout, _ = process.communicate()
        report = stdout
        logging.info(f"JasperGold for {sva_name} exited with code: {process.returncode}")

    except Exception as e:
        logging.error(f"Error running JasperGold for {sva_name}: {str(e)}")
        report = f"Error: {str(e)}\n"

    finally:
        if process and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                logging.info(f"Terminated running process for {sva_name}")
            except Exception as e:
                logging.error(f"Failed to terminate process for {sva_name}: {str(e)}")

    report_file_path = rpt_dir / f"{sva_name}.txt"
    report_file_path.write_text(report, encoding="utf-8")
    logging.info(f"Saved report to: {report_file_path}")
    return report_file_path

def run_ibex_verification(
    design_name: str,
    ibex_rtl_dir: Path,
    sva_dir: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    运行 Ibex if_stage 模块的验证。
    """
    verif_dir = output_dir / "verif"
    if verif_dir.exists():
        shutil.rmtree(verif_dir)
    verif_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. 解析 SVA 文件
        logging.info(f"Parsing SVA files from {sva_dir}")
        sva_files = [f for f in sva_dir.glob("*.sv") if f.is_file()]
        all_svas = parse_sva_from_json_files_structured(sva_files, "ibex_if_stage")
        
        total_sva_count = sum(len(v) for v in all_svas["ibex_if_stage"].values())
        logging.info(f"Found {total_sva_count} assertions for '{design_name}'.")
        
        if total_sva_count == 0:
            return {
                "task_status": "no_assertions",
                "assertion_status_counts": {"total": 0},
                "sva_details": {}
            }

        # 2. 创建文件列表
        file_list_path = verif_dir / "ibex_files.f"
        ibex_files = create_ibex_file_list(ibex_rtl_dir)
        
        with open(file_list_path, 'w', encoding='utf-8') as f:
            for file_path in ibex_files:
                f.write(f"{file_path}\n")
        
        logging.info(f"Created file list with {len(ibex_files)} files")

        # 3. 为每个断言创建验证环境
        ft_dir = verif_dir / "ft_rtl"
        tcl_dir = verif_dir / "tcl"
        jg_dir = verif_dir / "jg"
        rpt_dir = verif_dir / "rpt"
        
        for dir_path in [ft_dir, tcl_dir, jg_dir, rpt_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 获取目录路径
        prim_assert_dir = ibex_rtl_dir.parent / "vendor/lowrisc_ip/ip/prim/rtl"
        dv_utils_dir = ibex_rtl_dir.parent / "vendor/lowrisc_ip/dv/sv/dv_utils"
        rtl_dir = ibex_rtl_dir

        tcl_paths = []
        sva_mapping = {}

        # 为每个断言创建单独的验证
        for variable_name, assertions in all_svas["ibex_if_stage"].items():
            for sva_dict in assertions:
                sva_id = sva_dict["sva_id"]
                
                # 创建包含该断言的 RTL 文件
                ft_rtl_file = ft_dir / f"{sva_id}.sv"
                original_if_stage = ibex_rtl_dir / "ibex_if_stage.sv"
                
                if not insert_single_sva_into_module(
                    original_if_stage,
                    "ibex_if_stage",
                    sva_dict["sva_string"],
                    ft_rtl_file
                ):
                    logging.error(f"Failed to insert SVA {sva_id}")
                    continue

                # 创建文件列表（包含修改后的 if_stage）
                sva_file_list = verif_dir / f"{sva_id}.f"
                with open(sva_file_list, 'w', encoding='utf-8') as f:
                    for file_path in ibex_files:
                        if file_path.name == "ibex_if_stage.sv":
                            f.write(f"{ft_rtl_file}\n")  # 使用修改后的文件
                        else:
                            f.write(f"{file_path}\n")

                # 生成 TCL 脚本
                tcl_path = tcl_dir / f"{sva_id}.tcl"
                if generate_fpv_tcl(
                    tcl_path,
                    prim_assert_dir,
                    dv_utils_dir,
                    rtl_dir,
                    sva_file_list,
                    top_module="ibex_if_stage",
                    clock_signal="clk_i",
                    reset_signal="~rst_ni"
                ):
                    tcl_paths.append(tcl_path)
                    sva_mapping[tcl_path.stem] = sva_dict

        logging.info(f"Created {len(tcl_paths)} verification setups")

        # 4. 运行验证
        logging.info(f"Starting JasperGold runs for {len(tcl_paths)} assertions...")
        rpt_paths = []
        
        for tcl_path in tqdm(tcl_paths, desc="Running JasperGold"):
            rpt_path = run_jg_single(tcl_path, jg_dir, rpt_dir)
            rpt_paths.append(rpt_path)

        # 5. 分析结果
        logging.info("Analyzing verification results...")
        status_counts = {
            "proven": 0,
            "cex": 0, 
            "inconclusive": 0,
            "error": 0,
            "timeout": 0,
            "unknown": 0,
            "total": total_sva_count
        }

        # 更新每个断言的状态
        for rpt_path in rpt_paths:
            sva_id = rpt_path.stem
            if sva_id in sva_mapping:
                try:
                    report_content = rpt_path.read_text(encoding="utf-8")
                    status = extract_proof_status(report_content)
                    sva_mapping[sva_id]["status"] = status
                    status_counts[status] = status_counts.get(status, 0) + 1
                except Exception as e:
                    logging.error(f"Error reading report {rpt_path}: {e}")
                    sva_mapping[sva_id]["status"] = "error"
                    status_counts["error"] += 1

        # 6. 保存结果
        sva_status_file = verif_dir / "sva_status.json"
        with open(sva_status_file, 'w', encoding='utf-8') as f:
            json.dump(all_svas, f, indent=2, ensure_ascii=False)

        # 按变量统计结果
        variable_stats = {}
        for variable_name, assertions in all_svas["ibex_if_stage"].items():
            variable_stats[variable_name] = {
                "total": len(assertions),
                "proven": sum(1 for a in assertions if a.get("status") == "proven"),
                "cex": sum(1 for a in assertions if a.get("status") == "cex"),
                "inconclusive": sum(1 for a in assertions if a.get("status") == "inconclusive"),
                "error": sum(1 for a in assertions if a.get("status") == "error"),
            }

        # 确定总体状态
        if status_counts["proven"] == total_sva_count:
            task_status = "passed"
        elif status_counts["error"] == total_sva_count:
            task_status = "failed" 
        elif status_counts["proven"] > 0:
            task_status = "partial_pass"
        else:
            task_status = "inconclusive"

        result = {
            "task_status": task_status,
            "assertion_status_counts": status_counts,
            "variable_statistics": variable_stats,
            "sva_details": all_svas,
            "verification_directory": str(verif_dir)
        }

        logging.info(f"Verification completed. Task status: {task_status}")
        return result

    except Exception as e:
        logging.error(f"Error during verification: {e}", exc_info=True)
        return {
            "task_status": "failed",
            "error_message": str(e)
        }

def main():
    """
    主函数。
    """
    parser = argparse.ArgumentParser(description="Run Ibex if_stage verification")
    parser.add_argument("--design", default="ibex_if_stage", help="Design name")
    parser.add_argument("--ibex-dir", default=Path("/data/fhj/sva-var/ibex/rtl"), help="Path to Ibex RTL directory")
    parser.add_argument("--sva-dir", default=Path("/data/fhj/sva-var/verify/ibex_if_stage/data/assertions"), help="Path to SVA JSON files directory")
    parser.add_argument("--output-dir", default=Path("/data/fhj/sva-var/verify/ibex_if_stage/output"), help="Output directory")
    
    args = parser.parse_args()
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"ibex_verif_{timestamp}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 运行验证
    result = run_ibex_verification(
        design_name=args.design,
        ibex_rtl_dir=Path(args.ibex_dir),
        sva_dir=Path(args.sva_dir),
        output_dir=Path(args.output_dir)
    )
    
    # 保存最终结果
    result_file = Path(args.output_dir) / args.design / "verification_result.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Verification result saved to: {result_file}")

if __name__ == "__main__":
    main()