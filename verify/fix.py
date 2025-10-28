import json
import copy

def fix_mutation_errors(data):
    """
    修正突变测试结果中的错误统计
    """
    # 创建深拷贝以避免污染原数据
    fixed_data = copy.deepcopy(data)
    
    # 遍历所有变量结果
    for var_name, var_data in fixed_data["variable_results"].items():
        # 处理每个变量的details
        if "details" in var_data:
            fix_variable_details(var_data["details"])
            
            # 重新计算该变量的统计信息
            recalculate_variable_from_details(var_data)
    
    # 重新计算总体统计信息
    recalculate_overall_statistics(fixed_data)
    
    return fixed_data

def fix_variable_details(details):
    """
    修正变量级别details中每个具体突变的错误统计
    """
    for mut_key, mut_detail in details.items():
        if isinstance(mut_detail, dict):
            verification_status = mut_detail.get("verification_status", "")
            
            # 根据验证状态正确设置killed字段
            if verification_status == "cex":
                # cex代表杀死了变异体
                if not mut_detail.get("killed", False):
                    print(f"修正: {mut_key} - cex状态应该为killed: true")
                mut_detail["killed"] = True
                
            elif verification_status == "proven":
                # proven代表没杀死
                if mut_detail.get("killed", True):
                    print(f"修正: {mut_key} - proven状态应该为killed: false")
                mut_detail["killed"] = False
                
            elif verification_status == "error":
                # error代表验证失败，无法判断
                if mut_detail.get("killed", True):
                    print(f"修正: {mut_key} - error状态应该为killed: false")
                mut_detail["killed"] = False
                
                # 更新执行状态字段
                mut_detail["execution_status"] = "failed"

def recalculate_variable_from_details(var_data):
    """
    根据details重新计算变量级别的统计信息
    """
    if "details" not in var_data:
        return
        
    details = var_data["details"]
    
    # 重新计算突变类型统计
    mutation_type_stats = {}
    
    total_mutations = 0
    killed_mutations = 0
    error_mutations = 0
    
    # 首先按突变类型分组
    for mut_key, mut_detail in details.items():
        if isinstance(mut_detail, dict):
            mutation_type = mut_detail.get("mutation_type", "")
            
            if mutation_type not in mutation_type_stats:
                mutation_type_stats[mutation_type] = {
                    "total": 0,
                    "killed": 0,
                    "survived": 0,
                    "skipped": 0,
                    "error_count": 0
                }
            
            type_stat = mutation_type_stats[mutation_type]
            type_stat["total"] += 1
            total_mutations += 1
            
            verification_status = mut_detail.get("verification_status", "")
            
            if verification_status == "error":
                type_stat["error_count"] += 1
                error_mutations += 1
            elif mut_detail.get("skipped", False):
                type_stat["skipped"] += 1
            elif mut_detail.get("killed", False):
                type_stat["killed"] += 1
                killed_mutations += 1
            else:
                type_stat["survived"] += 1
    
    # 计算每个类型的杀死率
    for mutation_type, type_stat in mutation_type_stats.items():
        valid_mutations = type_stat["total"] - type_stat["error_count"] - type_stat["skipped"]
        if valid_mutations > 0:
            type_stat["killed_rate"] = type_stat["killed"] / valid_mutations
        else:
            type_stat["killed_rate"] = 0.0
    
    # 更新变量数据
    var_data["mutation_type_statistics"] = mutation_type_stats
    var_data["total_mutations"] = total_mutations
    var_data["killed_mutations"] = killed_mutations
    
    # 计算变量级别的变异分数（基于有效突变体）
    valid_mutations = total_mutations - error_mutations
    if valid_mutations > 0:
        var_data["mutation_score"] = killed_mutations / valid_mutations
    else:
        var_data["mutation_score"] = 0.0

def recalculate_overall_statistics(data):
    """
    重新计算总体统计信息
    """
    total_variables = 0
    total_assertions = 0
    total_mutations_tested = 0
    total_killed_mutations = 0
    total_valid_mutations = 0
    total_error_mutations = 0
    
    # 遍历所有变量
    for var_name, var_data in data["variable_results"].items():
        total_variables += 1
        total_assertions += var_data.get("assertion_count", 0)
        total_mutations_tested += var_data.get("total_mutations", 0)
        total_killed_mutations += var_data.get("killed_mutations", 0)
        
        # 计算错误突变体数量
        var_error_mutations = 0
        if "details" in var_data:
            for mut_key, mut_detail in var_data["details"].items():
                if (isinstance(mut_detail, dict) and 
                    mut_detail.get("verification_status") == "error"):
                    var_error_mutations += 1
        
        total_error_mutations += var_error_mutations
    
    # 计算有效突变体
    total_valid_mutations = total_mutations_tested - total_error_mutations
    
    # 更新总体统计
    data["total_variables"] = total_variables
    data["total_assertions"] = total_assertions
    data["total_mutations_tested"] = total_mutations_tested
    data["total_killed_mutations"] = total_killed_mutations
    data["total_valid_mutations"] = total_valid_mutations
    data["total_error_mutations"] = total_error_mutations
    
    # 计算总体变异分数（基于有效突变体）
    if total_valid_mutations > 0:
        data["overall_mutation_score"] = total_killed_mutations / total_valid_mutations
    else:
        data["overall_mutation_score"] = 0.0

def main():
    # 读取原始JSON文件
    input_file = "/data/fhj/sva-var/verify/ibex_if_stage/mutation_testing_results.json"  # 请修改为您的实际文件名
    output_file = "/data/fhj/sva-var/verify/ibex_if_stage/mutation_results_fixed.json"
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        print(f"正在处理文件: {input_file}")
        print(f"原始总体变异分数: {original_data['overall_mutation_score']}")
        
        # 修正数据
        fixed_data = fix_mutation_errors(original_data)
        
        # 保存修正后的文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fixed_data, f, indent=2, ensure_ascii=False)
        
        print(f"修正完成! 结果已保存到: {output_file}")
        print(f"修正后总体变异分数: {fixed_data['overall_mutation_score']}")
        print(f"总突变体数量: {fixed_data['total_mutations_tested']}")
        print(f"有效突变体数量: {fixed_data['total_valid_mutations']}")
        print(f"错误突变体数量: {fixed_data['total_error_mutations']}")
        
        # 显示修正摘要
        print("\n修正摘要:")
        total_fixed = 0
        for var_name, var_data in fixed_data["variable_results"].items():
            if "details" in var_data:
                for mut_key, mut_detail in var_data["details"].items():
                    if (isinstance(mut_detail, dict) and 
                        mut_detail.get("verification_status") == "error" and
                        mut_detail.get("execution_status") == "failed"):
                        total_fixed += 1
        print(f"共修正了 {total_fixed} 个验证错误的突变体")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except json.JSONDecodeError as e:
        print(f"错误: JSON文件格式错误 - {e}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()