import re
import json
from pathlib import Path
from typing import List, Dict, Any
import networkx as nx

def process_files_in_order(pre_verilog_files: List[Path], output_json: Path = None):
    """
    Analyzes Verilog/SystemVerilog files to determine the correct compilation order.
    It handles `include` directives, module instantiations, and prioritizes macro definition files.
    """
    verilog_files = []
    macro_files = [] # 专门存放宏定义文件

    print("Scanning for Verilog/SystemVerilog files...")
    for file in pre_verilog_files:
        if str(file).endswith((".v", ".sv", ".vh")):
            verilog_files.append(file.resolve())
            # 启发式规则：.vh文件或包含"define"的文件名很可能是宏文件
            if str(file).endswith(".vh") or "define" in str(file).lower():
                macro_files.append(file.resolve())
                print(f"Identified potential macro file: {file.name}")

    if not verilog_files:
        print("No Verilog/SystemVerilog files found.")
        return []

    # --- 数据结构初始化 ---
    include_map = {}
    module_map = {}
    file_contents = {}
    verilog_keywords = {
        "module", "macromodule", "task", "function", "class", "interface",
        "program", "package", "primitive", "config", "property", "sequence",
        "always", "always_comb", "always_ff", "always_latch", "initial", "assign",
        "if", "else", "case", "casex", "casez", "for", "forever", "repeat", "while",
        "begin", "end", "fork", "join","generate"
    }

    # --- 第一遍扫描：读取内容，识别模块定义和`include` ---
    for file_path in verilog_files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            file_contents[file_path] = content

            # 移除注释，简化后续解析
            content_no_comments = re.sub(r'//.*', '', content)
            content_no_comments = re.sub(r'/\*.*?\*/', '', content_no_comments, flags=re.DOTALL)

            # 提取 `include`
            # 使用 os.path.normpath 和 os.path.join 来正确处理相对路径
            includes = [
                (Path(file_path.parent) / match.group(1)).resolve()
                for match in re.finditer(r'`include\s+"([^"]+)"', content_no_comments)
            ]
            include_map[file_path] = includes

            # 提取模块定义
            for match in re.finditer(r'\bmodule\s+([a-zA-Z_][\w]*)', content_no_comments):
                module_name = match.group(1)
                if module_name not in module_map:
                    module_map[module_name] = file_path
                else:
                    print(f"Duplicate module definition for '{module_name}' found in {file_path.name} and {module_map[module_name].name}. Using the first one found.")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    # --- 第二遍：构建依赖图 ---
    dep_graph = nx.DiGraph()
    for file_path in verilog_files:
        dep_graph.add_node(file_path)

    # 收集依赖关系详细信息
    dependency_details = {
        "include_dependencies": [],
        "module_dependencies": [],
        "all_dependencies": []
    }

    # 添加 `include` 依赖 (file_path `include`s an included_file, so file_path depends on included_file)
    for file_path, includes in include_map.items():
        for included_file in includes:
            if included_file in dep_graph:
                dep_graph.add_edge(file_path, included_file)
                dependency_details["include_dependencies"].append({
                    "source_file": str(file_path.name),
                    "source_path": str(file_path),
                    "target_file": str(included_file.name),
                    "target_path": str(included_file),
                    "dependency_type": "include",
                    "description": f"File {file_path.name} includes {included_file.name} via `include directive"
                })

    # 添加模块实例化依赖 (file_path instantiates a module, so file_path depends on the file defining that module)
    for file_path, content in file_contents.items():
        content_no_comments = re.sub(r'//.*', '', content)
        content_no_comments = re.sub(r'/\*.*?\*/', '', content_no_comments, flags=re.DOTALL)

        # **关键改进**：更精确的实例化匹配，排除了关键字
        inst_pattern = r'\b([a-zA-Z_][\w]*)\s*(?:#\s*\(.*?\))?\s+([a-zA-Z_][\w]*)\s*\('
        for match in re.finditer(inst_pattern, content_no_comments):
            module_type, instance_name = match.groups()

            # 如果匹配到的"模块名"是关键字，则跳过
            if module_type in verilog_keywords:
                continue

            if module_type in module_map:
                defining_file = module_map[module_type]
                if defining_file != file_path:
                    dep_graph.add_edge(file_path, defining_file)
                    dependency_details["module_dependencies"].append({
                        "source_file": str(file_path.name),
                        "source_path": str(file_path),
                        "target_file": str(defining_file.name),
                        "target_path": str(defining_file),
                        "module_type": module_type,
                        "instance_name": instance_name,
                        "dependency_type": "module_instantiation",
                        "description": f"File {file_path.name} instantiates module '{module_type}' (instance '{instance_name}') defined in {defining_file.name}"
                    })
            else:
                print(f"Undefined module '{module_type}' (instance '{instance_name}') referenced in {file_path.name}")

    # 合并所有依赖关系
    dependency_details["all_dependencies"] = (
        dependency_details["include_dependencies"] + 
        dependency_details["module_dependencies"]
    )

    # --- 第三遍：排序 ---
    try:
        # 拓扑排序给出了一个依赖顺序 (依赖项在前)
        # 我们需要反过来，被依赖的（基础模块）在前
        sorted_files = list(reversed(list(nx.topological_sort(dep_graph))))

        # **关键改进**：将所有宏文件强制移动到列表的最前面
        final_sorted_list = []
        # 先添加所有宏文件，并从主列表中移除
        for mf in macro_files:
            if mf in sorted_files:
                final_sorted_list.append(mf)
                sorted_files.remove(mf)

        # 然后添加剩余的、已经排好序的文件
        final_sorted_list.extend(sorted_files)

        print("Determined processing order:")
        for i, file_path in enumerate(final_sorted_list, 1):
            print(f"{i}. {file_path.name}")

        # 生成并输出JSON依赖关系文件
        if output_json:
            generate_dependency_json(
                final_sorted_list, 
                dependency_details, 
                module_map, 
                include_map,
                output_json
            )

        return final_sorted_list

    except nx.NetworkXUnfeasible:
        print("Cyclic dependency detected in the design, cannot determine a valid compilation order.")
        try:
            cycle = nx.find_cycle(dep_graph)
            cycle_info = " -> ".join([p.name for p, _ in cycle] + [cycle[0][0].name])
            print("Cycle found: " + cycle_info)
            
            # 即使有循环依赖，也生成JSON文件
            if output_json:
                generate_dependency_json(
                    sorted(verilog_files, key=lambda p: 0 if p in macro_files else 1),
                    dependency_details, 
                    module_map, 
                    include_map,
                    output_json,
                    has_cycle=True,
                    cycle_info=cycle_info
                )
        except nx.NetworkXNoCycle:
            pass
        # 在有循环的情况下，返回一个尽力而为的列表（宏文件优先）
        return sorted(verilog_files, key=lambda p: 0 if p in macro_files else 1)

def generate_dependency_json(
    sorted_files: List[Path],
    dependency_details: Dict[str, Any],
    module_map: Dict[str, Path],
    include_map: Dict[Path, List[Path]],
    output_path: Path,
    has_cycle: bool = False,
    cycle_info: str = None
):
    """生成依赖关系的JSON文件"""
    
    # 构建文件信息
    file_info = {}
    for file_path in set(sorted_files):
        file_info[str(file_path)] = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "is_macro_file": file_path.name.endswith(".vh") or "define" in file_path.name.lower(),
            "defined_modules": [
                module for module, path in module_map.items() if path == file_path
            ],
            "included_files": [
                str(inc_file.name) for inc_file in include_map.get(file_path, [])
            ]
        }
    
    # 构建编译顺序
    compilation_order = [
        {
            "order": i + 1,
            "file_name": file_path.name,
            "file_path": str(file_path),
            "is_macro_file": file_path.name.endswith(".vh") or "define" in file_path.name.lower()
        }
        for i, file_path in enumerate(sorted_files)
    ]
    
    # 构建完整的JSON数据结构
    json_data = {
        "dependency_analysis": {
            "summary": {
                "total_files": len(sorted_files),
                "total_dependencies": len(dependency_details["all_dependencies"]),
                "include_dependencies": len(dependency_details["include_dependencies"]),
                "module_dependencies": len(dependency_details["module_dependencies"]),
                "has_cyclic_dependencies": has_cycle,
                "cycle_info": cycle_info if has_cycle else None,
                "description": "Verilog/SystemVerilog文件依赖关系分析，包括include依赖和模块实例化依赖"
            },
            "compilation_order": compilation_order,
            "file_details": file_info,
            "dependency_relationships": dependency_details,
            "module_definitions": {
                module: {
                    "defined_in": str(path),
                    "defined_in_file": path.name
                }
                for module, path in module_map.items()
            }
        }
    }
    
    # 写入JSON文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"Dependency analysis JSON saved to: {output_path}")
    except Exception as e:
        print(f"Error saving dependency JSON file: {e}")

# 修改analyze_design函数以支持JSON输出
def analyze_design(verilog_files: List[Path], output_dir: Path, dependency_json: str = "dependency_analysis.json"):
    """Main method to analyze all Verilog files and combine them in dependency order."""
    print(f"Analyzing RTL design in directory:{verilog_files}")

    # 设置JSON输出路径
    json_output_path = output_dir / dependency_json

    # 使用重构后的函数获取排序后的文件列表，并生成JSON
    sorted_verilog_files = process_files_in_order(verilog_files, json_output_path)

    if not sorted_verilog_files:
        print("Analysis failed: No files to process or dependency resolution failed.")
        return

    print(f"Combining {len(sorted_verilog_files)} files into a single file...")

    combined_content = ""
    # **关键改进**：因为 `include` 关系已经在排序中体现，我们不再需要移除 `include` 语句。
    # 合并时保留 `include` 可以更好地处理宏。
    for file_path in sorted_verilog_files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            combined_content += f"\n// ----- Start of File: {file_path.name} -----\n"
            combined_content += content
            combined_content += f"\n// ----- End of File: {file_path.name} -----\n"
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    combined_content = re.sub(r'`include\s+"[^"]+"\s*', "", combined_content)
    combined_file = output_dir / "_combined_rtl.sv"
    combined_file_no_comments = output_dir / "_combined_rtl_no_comments.sv"

    try:
        combined_file.write_text(combined_content, encoding="utf-8")
        print(f"Created combined RTL file: {combined_file}")
        # 注意：这里需要确保remove_comments函数存在
        # remove_comments(combined_file, f"{output_dir}/_combined_rtl_no_comments.sv")
        print(f"Created combined RTL file with no comments: {combined_file_no_comments}")

    except Exception as e:
        print(f"Error creating combined file: {e}")