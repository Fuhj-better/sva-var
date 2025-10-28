#!/usr/bin/env python3
"""
变量特征权重优化工具 - 修复并优化版本
目标：通过最大化Spearman/Kendall相关系数，拟合特征权重，以更好地预测变量的变异分数排序。
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt

# --- 数据加载和预处理（保持原样，但移除不稳定的不敏感匹配） ---

def load_data(groundtruth_file: Path, features_file: Path) -> tuple:
    """
    加载groundtruth和特征数据。
    注意：为提高健壮性，移除了大小写不敏感匹配，因为硬件变量名通常是大小写敏感的。
    """
    print("Loading groundtruth file...")
    with open(groundtruth_file, 'r', encoding='utf-8') as f:
        mutation_results = json.load(f)
    
    print("Loading features file...")
    with open(features_file, 'r', encoding='utf-8') as f:
        features_data = json.load(f)
    
    variable_results = mutation_results.get("variable_results", {})
    groundtruth_scores = {
        var_name: result.get("mutation_score", 0)
        for var_name, result in variable_results.items()
        if "error" not in result
    }
    
    print(f"Found {len(groundtruth_scores)} variables in groundtruth")
    
    feature_names = [
        'Data_Dependency_Centrality',
        'Control_Coupling_Strength', 
        'Propagation_Risk',
        'Contextual_Complexity',
        'Substitutability',
        'Controller_Dependence',
        'Functional_Contribution',
        'Safety_Sensitivity'
    ]
    
    features = {}
    matched_count = 0
    
    # 仅使用精确匹配
    for var_name, feature_dict in features_data.items():
        if var_name in groundtruth_scores:
            feature_vector = [feature_dict.get(feature, 0) for feature in feature_names]
            features[var_name] = feature_vector
            matched_count += 1
            
    # 筛选出只有特征数据的变量
    final_groundtruth = {var: groundtruth_scores[var] for var in features.keys()}
    
    print(f"Successfully matched {matched_count} variables for features and groundtruth")
    
    return final_groundtruth, features, feature_names

def calculate_predicted_scores(features: Dict, weights: np.ndarray) -> Dict:
    """计算预测得分。"""
    predicted_scores = {}
    for var_name, feature_vector in features.items():
        # 线性加权组合 (weights 已在优化器中被约束为正且和为1)
        score = np.dot(feature_vector, weights)
        predicted_scores[var_name] = score
    return predicted_scores

# --- 核心优化函数（关键修改） ---

def ranking_loss(weights: np.ndarray, groundtruth_scores: Dict, features: Dict, 
                 metric: str = 'spearman') -> float:
    """
    计算排序损失函数（1 - 相关性）。
    注意：此版本假设传入的weights已满足约束（和为1，非负）。
    """
    # 无需再次归一化，依赖优化器的约束
    predicted_scores = calculate_predicted_scores(features, weights)
    common_vars = list(predicted_scores.keys())
    
    if len(common_vars) < 2:
        return 1.0
    
    groundtruth_ranks = [groundtruth_scores[var] for var in common_vars]
    predicted_ranks = [predicted_scores[var] for var in common_vars]
    
    # 计算排序相关性
    if metric == 'spearman':
        correlation, _ = spearmanr(groundtruth_ranks, predicted_ranks)
    elif metric == 'kendall':
        correlation, _ = kendalltau(groundtruth_ranks, predicted_ranks)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    if np.isnan(correlation):
        return 1.0
    
    # 损失是 1 - 相关性
    return 1.0 - correlation

def optimize_weights(groundtruth_scores: Dict, features: Dict, feature_names: List[str], 
                     metric: str = 'spearman') -> Dict:
    """
    优化特征权重。
    关键修改：尝试使用更鲁棒的优化方法，并简化约束。
    """
    n_features = len(feature_names)
    
    # 初始权重（均匀分布）
    initial_weights = np.ones(n_features) / n_features
    
    # 1. 定义边界 (Bounds): 权重必须是非负数 (0 <= x_i <= 1)
    bounds = Bounds(0, 1)
    
    # 2. 定义线性约束 (LinearConstraint): 权重之和必须为 1 (sum(x_i) = 1)
    # 约束矩阵 A: [1, 1, ..., 1]
    # 约束范围 L <= A @ x <= U: [1] <= A @ x <= [1]
    linear_constraint = LinearConstraint(np.ones(n_features), [1], [1])
    
    print(f"Starting optimization with {len(features)} variables and {n_features} features...")
    
    # 尝试多种优化方法，以提高成功率
    methods_to_try = ['SLSQP', 'L-BFGS-B', 'Nelder-Mead'] 
    
    optimal_weights = initial_weights
    
    for method in methods_to_try:
        print(f"--- Trying optimization method: {method} ---")
        
        # 只有SLSQP和Trust-constr支持Bounds和LinearConstraint
        # L-BFGS-B只支持Bounds，我们需要自己处理约束（通过惩罚项或重写损失函数，但这里SLSQP更合适）
        # Nelder-Mead不支持约束，但可以用于探索性搜索
        
        if method == 'SLSQP':
            result = minimize(
                ranking_loss,
                initial_weights,
                args=(groundtruth_scores, features, metric),
                method='SLSQP',
                bounds=bounds,
                constraints=linear_constraint,
                options={'maxiter': 5000, 'ftol': 1e-6} # 增加迭代次数
            )
        elif method == 'L-BFGS-B':
            # L-BFGS-B 不支持等式约束，跳过
            continue
        elif method == 'Nelder-Mead':
            # Nelder-Mead 不支持显式约束，需要修改损失函数处理归一化，
            # 为了保持代码简洁和约束明确，这里暂时跳过，只尝试SLSQP。
            # 如果SLSQP失败，就直接使用初始权重。
             continue

        if result.success or ranking_loss(result.x, groundtruth_scores, features, metric) < ranking_loss(optimal_weights, groundtruth_scores, features, metric):
            
            # 确保结果非负且归一化 (尽管SLSQP应该已处理)
            candidate_weights = np.abs(result.x)
            if np.sum(candidate_weights) > 0:
                candidate_weights = candidate_weights / np.sum(candidate_weights)
            
            # 检查相关性是否真的提高了
            current_loss = ranking_loss(optimal_weights, groundtruth_scores, features, metric)
            candidate_loss = ranking_loss(candidate_weights, groundtruth_scores, features, metric)
            
            if candidate_loss < current_loss:
                optimal_weights = candidate_weights
                print(f"Optimization successful with {method}. New Loss: {candidate_loss:.4f}")
                # 找到更好的结果，可以提前退出
                break 
            else:
                 print(f"Optimization finished with {method}, but did not improve ranking loss.")
        else:
            print(f"Optimization failed with {method}: {result.message}")
            
    
    # 如果所有尝试都失败或没有改进，则使用初始权重
    if np.array_equal(optimal_weights, initial_weights):
        print("Final optimization failed or found no improvement, defaulting to initial uniform weights.")
        
    return optimal_weights

# --- 评估和可视化（小幅清理和改进） ---

def evaluate_weights(groundtruth_scores: Dict, features: Dict, weights: np.ndarray, 
                     feature_names: List[str]) -> Dict:
    """评估权重效果。"""
    # 确保权重归一化，以防优化器返回非归一化的结果
    weights = np.abs(weights)
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    
    # ... (其余部分保持不变)
    
    # 计算预测得分
    predicted_scores = calculate_predicted_scores(features, weights)
    
    # 计算各种指标
    common_vars = list(predicted_scores.keys())
    # ... (省略重复代码)
    
    groundtruth_values = [groundtruth_scores[var] for var in common_vars]
    predicted_values = [predicted_scores[var] for var in common_vars]
    
    # 排序相关性
    if len(common_vars) >= 2:
        spearman_corr, _ = spearmanr(groundtruth_values, predicted_values)
        kendall_corr, _ = kendalltau(groundtruth_values, predicted_values)
    else:
        spearman_corr = 0
        kendall_corr = 0
    
    # 创建排序比较
    groundtruth_ranking = sorted(common_vars, key=lambda x: groundtruth_scores[x], reverse=True)
    predicted_ranking = sorted(common_vars, key=lambda x: predicted_scores[x], reverse=True)
    
    # 计算前N个的重合度
    def top_n_overlap(rank1, rank2, n):
        if len(rank1) < n or len(rank2) < n:
            return 0
        set1 = set(rank1[:n])
        set2 = set(rank2[:n])
        return len(set1.intersection(set2)) / n
    
    top_5_overlap = top_n_overlap(groundtruth_ranking, predicted_ranking, 5)
    top_10_overlap = top_n_overlap(groundtruth_ranking, predicted_ranking, 10)
    top_20_overlap = top_n_overlap(groundtruth_ranking, predicted_ranking, 20)
    
    return {
        'spearman_correlation': spearman_corr if not np.isnan(spearman_corr) else 0,
        'kendall_correlation': kendall_corr if not np.isnan(kendall_corr) else 0,
        'top_5_overlap': top_5_overlap,
        'top_10_overlap': top_10_overlap,
        'top_20_overlap': top_20_overlap,
        'groundtruth_ranking': groundtruth_ranking,
        'predicted_ranking': predicted_ranking,
        'predicted_scores': predicted_scores,
        'groundtruth_scores': {var: groundtruth_scores[var] for var in common_vars}
    }

def create_visualizations(evaluation_results: Dict, weights: np.ndarray, 
                          feature_names: List[str], output_dir: Path):
    """创建可视化图表。"""
    # ... (保持原样，功能已完善)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(evaluation_results['groundtruth_ranking']) == 0:
        print("No data for visualization")
        return
    
    # 1. 权重分布图
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(weights)), weights, color='skyblue', alpha=0.7)
    ax.set_xlabel('Features')
    ax.set_ylabel('Weight')
    ax.set_title('Optimized Feature Weights')
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_weights.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 排序相关性散点图
    if len(evaluation_results['groundtruth_ranking']) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 确保数据点顺序一致
        common_vars = evaluation_results['predicted_ranking']
        groundtruth_scores_list = [evaluation_results['groundtruth_scores'][var] for var in common_vars]
        predicted_scores_list = [evaluation_results['predicted_scores'][var] for var in common_vars]
        
        scatter = ax.scatter(groundtruth_scores_list, predicted_scores_list, alpha=0.6, s=50)
        ax.set_xlabel('Groundtruth Mutation Score')
        ax.set_ylabel('Predicted Score')
        ax.set_title(f'Predicted vs Groundtruth Scores\n(Spearman r = {evaluation_results["spearman_correlation"]:.3f})')
        
        # 添加趋势线
        if len(groundtruth_scores_list) >= 2:
            z = np.polyfit(groundtruth_scores_list, predicted_scores_list, 1)
            p = np.poly1d(z)
            ax.plot(groundtruth_scores_list, p(groundtruth_scores_list), "r--", alpha=0.8)
        
        # 为一些点添加标签
        for i, var in enumerate(common_vars[:min(10, len(common_vars))]):
            ax.annotate(var, (groundtruth_scores_list[i], predicted_scores_list[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

# --- 主函数（保持原样） ---

def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="Optimize feature weights based on mutation testing groundtruth")
    parser.add_argument("--groundtruth-file", default=Path("/data/fhj/sva-var/verify/ibex_if_stage/mutation_results_fixed.json"), type=Path,
                        help="Path to mutation_testing_results.json file")
    parser.add_argument("--features-file", default=Path("/data/fhj/sva-var/verify/ibex_if_stage/data/vars/vars.json"), type=Path,
                        help="Path to features JSON file")
    parser.add_argument("--output-dir", type=Path, default=Path("./ibex_if_stage/weight_optimization"),
                        help="Output directory for results")
    parser.add_argument("--metric", choices=['spearman', 'kendall'], default='spearman',
                        help="Correlation metric to optimize")
    
    args = parser.parse_args()
    
    if not args.groundtruth_file.exists():
        print(f"Error: Groundtruth file not found: {args.groundtruth_file}")
        return
    if not args.features_file.exists():
        print(f"Error: Features file not found: {args.features_file}")
        return
    
    # 加载数据
    groundtruth_scores, features, feature_names = load_data(args.groundtruth_file, args.features_file)
    
    if len(features) < 2:
        print(f"Error: Found only {len(features)} matching variables. Need at least 2 for ranking optimization.")
        return
    
    print(f"Using {len(features)} variables for optimization")
    
    # 优化权重
    optimal_weights = optimize_weights(groundtruth_scores, features, feature_names, args.metric)
    
    # 评估结果
    evaluation_results = evaluate_weights(groundtruth_scores, features, optimal_weights, feature_names)
    
    # 打印结果
    print("\n" + "="*60)
    print("WEIGHT OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"\nOptimized Weights:")
    for feature, weight in zip(feature_names, optimal_weights):
        print(f"  {feature:30}: {weight:.4f}")
    
    print(f"\nEvaluation Metrics:")
    print(f"  Spearman Correlation: {evaluation_results['spearman_correlation']:.4f}")
    print(f"  Kendall Correlation:  {evaluation_results['kendall_correlation']:.4f}")
    print(f"  Top 5 Overlap:       {evaluation_results['top_5_overlap']:.2%}")
    print(f"  Top 10 Overlap:      {evaluation_results['top_10_overlap']:.2%}")
    print(f"  Top 20 Overlap:      {evaluation_results['top_20_overlap']:.2%}")
    
    if evaluation_results['groundtruth_ranking']:
        print(f"\nTop 10 Variables Comparison:")
        print(f"{'Rank':<4} {'Groundtruth':<20} {'Predicted':<20} {'Mutation Score':<12} {'Predicted Score':<15}")
        print("-" * 80)
        for i in range(min(10, len(evaluation_results['groundtruth_ranking']))):
            gt_var = evaluation_results['groundtruth_ranking'][i]
            pred_var = evaluation_results['predicted_ranking'][i]
            gt_score = groundtruth_scores[gt_var]
            pred_score = evaluation_results['predicted_scores'][pred_var]
            print(f"{i+1:<4} {gt_var:<20} {pred_var:<20} {gt_score:>11.1%} {pred_score:>14.3f}")
    
    # 保存结果
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存权重和评估结果
    results = {
        'optimal_weights': dict(zip(feature_names, optimal_weights.tolist())),
        'evaluation_metrics': {
            'spearman_correlation': evaluation_results['spearman_correlation'],
            'kendall_correlation': evaluation_results['kendall_correlation'],
            'top_5_overlap': evaluation_results['top_5_overlap'],
            'top_10_overlap': evaluation_results['top_10_overlap'],
            'top_20_overlap': evaluation_results['top_20_overlap']
        },
        'groundtruth_ranking': evaluation_results['groundtruth_ranking'],
        'predicted_ranking': evaluation_results['predicted_ranking'],
        'predicted_scores': evaluation_results['predicted_scores']
    }
    
    with open(args.output_dir / 'optimization_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 创建可视化
    print("\nCreating visualizations...")
    create_visualizations(evaluation_results, optimal_weights, feature_names, args.output_dir / "visualizations")
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()