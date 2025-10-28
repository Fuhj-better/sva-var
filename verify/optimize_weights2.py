#!/usr/bin/env python3
"""
变量特征权重优化工具 - 基于图像逻辑回归修复版本
目标：通过最小化二元交叉熵损失，拟合特征权重 w 和偏置 b，以预测变量的 Top N 重要性（二元分类）。
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt

# --- 数据加载和预处理 (修改: 创建二元 Groundtruth) ---

def load_data(groundtruth_file: Path, features_file: Path, top_n: int = 20) -> tuple:
    """
    加载groundtruth和特征数据，并将连续分数转换为二元标签 z (Top N 为 1，其余为 0)。
    """
    print("Loading groundtruth file...")
    with open(groundtruth_file, 'r', encoding='utf-8') as f:
        mutation_results = json.load(f)
    
    print("Loading features file...")
    with open(features_file, 'r', encoding='utf-8') as f:
        features_data = json.load(f)
    
    variable_results = mutation_results.get("variable_results", {})
    groundtruth_scores_continuous = {
        var_name: result.get("mutation_score", 0)
        for var_name, result in variable_results.items()
        if "error" not in result
    }
    
    print(f"Found {len(groundtruth_scores_continuous)} variables in groundtruth")
    
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
    
    # 仅使用精确匹配
    for var_name, feature_dict in features_data.items():
        if var_name in groundtruth_scores_continuous:
            # 特征向量 x
            feature_vector = [feature_dict.get(feature, 0) for feature in feature_names]
            features[var_name] = np.array(feature_vector)
            
    # 筛选出只有特征数据的变量
    final_scores = {var: groundtruth_scores_continuous[var] for var in features.keys()}
    
    print(f"Successfully matched {len(final_scores)} variables for features and groundtruth")
    
    # --- 关键修改：创建二元 Groundtruth z ---
    
    if len(final_scores) < top_n:
        # 如果变量不足 top_n，则全部标记为 1 (如果非空)
        top_n_vars = list(final_scores.keys())
        if not top_n_vars:
            print("Error: No variables found.")
            return {}, {}, [], 0
    else:
        # 按分数降序排列，取前 top_n 个
        sorted_vars = sorted(final_scores, key=final_scores.get, reverse=True)
        top_n_vars = set(sorted_vars[:top_n])

    # 创建二元标签 z: Top N 为 1，其余为 0
    groundtruth_labels_binary = {
        var: 1 if var in top_n_vars else 0
        for var in final_scores.keys()
    }
    
    print(f"Binary target z created: {len(top_n_vars)} variables labeled as 1.")
    
    # 返回二元标签作为新的 groundtruth
    return groundtruth_labels_binary, features, feature_names, top_n

# --- 核心预测和损失函数（完全重写） ---

def sigmoid(t: np.ndarray) -> np.ndarray:
    """Sigmoid 激活函数: sigma(t) = 1 / (1 + e^-t)"""
    return 1.0 / (1.0 + np.exp(-t))

def predict_z(features: Dict, params: np.ndarray, feature_names: List[str]) -> Dict:
    """
    计算预测得分 z_hat = sigma(w^T x + b)。
    params = [w_1, w_2, ..., w_d, b]
    """
    n_features = len(feature_names)
    weights = params[:-1]  # w_x (d 维)
    bias = params[-1]      # b (1 维)
    
    predicted_scores = {}
    for var_name, feature_vector in features.items():
        # w^T x + b
        linear_combination = np.dot(feature_vector, weights) + bias
        # sigma(w^T x + b)
        score = sigmoid(linear_combination)
        predicted_scores[var_name] = score
        
    return predicted_scores

def binary_cross_entropy_loss(params: np.ndarray, groundtruth_labels: Dict, features: Dict, 
                             feature_names: List[str]) -> float:
    """
    计算二元交叉熵损失 (Binary Cross-Entropy Loss)。
    L = -1/N * sum [z * log(z_hat) + (1-z) * log(1-z_hat)]
    """
    predicted_scores = predict_z(features, params, feature_names)
    common_vars = list(predicted_scores.keys())
    
    if len(common_vars) < 1:
        return 1e6 # 返回一个很大的损失值
    
    groundtruth_z = np.array([groundtruth_labels[var] for var in common_vars])
    predicted_z_hat = np.array([predicted_scores[var] for var in common_vars])
    
    # 避免 log(0)，对预测值进行裁剪
    epsilon = 1e-15
    predicted_z_hat = np.clip(predicted_z_hat, epsilon, 1 - epsilon)
    
    # 计算 BCE 损失
    loss = -np.mean(groundtruth_z * np.log(predicted_z_hat) + (1 - groundtruth_z) * np.log(1 - predicted_z_hat))
    
    if np.isnan(loss) or loss > 1e6:
        return 1e6
    
    return loss

def optimize_logistic_regression_weights(groundtruth_labels: Dict, features: Dict, 
                                       feature_names: List[str]) -> np.ndarray:
    """
    优化 Logistic Regression 的特征权重 w 和偏置 b。
    """
    n_features = len(feature_names)
    # 参数数量 = 特征数 + 1 (偏置 b)
    n_params = n_features + 1 
    
    # 初始参数（权重和偏置，全部设为 0）
    initial_params = np.zeros(n_params)
    
    # 边界 (Bounds): 对权重和偏置不设强制限制，使用 None 或较大的范围
    # Logistic Regression 不要求权重非负或归一化
    bounds = Bounds([-np.inf] * n_params, [np.inf] * n_params)
    
    print(f"Starting Logistic Regression optimization with {len(features)} variables and {n_params} parameters...")
    
    # 使用 L-BFGS-B 或 SLSQP，它们适用于带有 L2 惩罚项（虽然这里没有）或非线性函数的优化
    # L-BFGS-B 通常是一个很好的起点
    result = minimize(
        binary_cross_entropy_loss,
        initial_params,
        args=(groundtruth_labels, features, feature_names),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-6}
    )
    
    if result.success:
        print(f"Optimization successful. Final BCE Loss: {result.fun:.4f}")
    else:
        print(f"Optimization failed: {result.message}. Using initial parameters.")
        
    return result.x if result.success else initial_params

# --- 评估和可视化 (修改: 评估二元分类指标) ---

def evaluate_results(groundtruth_labels: Dict, features: Dict, params: np.ndarray, 
                     feature_names: List[str], top_n: int) -> Dict:
    """评估 Logistic Regression 的效果。"""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    # 计算预测得分 z_hat
    predicted_scores = predict_z(features, params, feature_names)
    
    common_vars = list(predicted_scores.keys())
    
    groundtruth_z = np.array([groundtruth_labels[var] for var in common_vars])
    predicted_z_hat = np.array([predicted_scores[var] for var in common_vars])
    
    # 将 z_hat 转换为二元预测 (通常阈值为 0.5)
    predicted_z_binary = (predicted_z_hat >= 0.5).astype(int)
    
    if len(common_vars) >= 2:
        acc = accuracy_score(groundtruth_z, predicted_z_binary)
        f1 = f1_score(groundtruth_z, predicted_z_binary)
        precision = precision_score(groundtruth_z, predicted_z_binary)
        recall = recall_score(groundtruth_z, predicted_z_binary)
        try:
            auc_roc = roc_auc_score(groundtruth_z, predicted_z_hat)
        except ValueError:
            auc_roc = 0.0 # 样本全为一类时 AUC 无法计算
    else:
        acc, f1, precision, recall, auc_roc = 0.0, 0.0, 0.0, 0.0, 0.0

    # 仍然计算 Top N 重合度 (将预测分数作为排名依据)
    groundtruth_ranking = sorted(common_vars, key=lambda x: groundtruth_labels[x], reverse=True)
    predicted_ranking = sorted(common_vars, key=lambda x: predicted_scores[x], reverse=True)

    def top_n_overlap(rank1, rank2, n):
        # 使用 top_n 变量作为重要性基准，而不是分类结果
        n = min(n, len(rank1), len(rank2))
        if n == 0: return 0
        set1 = set(rank1[:n])
        set2 = set(rank2[:n])
        return len(set1.intersection(set2)) / n
    
    top_n_overlap_result = top_n_overlap(groundtruth_ranking, predicted_ranking, top_n)
    
    return {
        'accuracy': acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        f'top_{top_n}_overlap': top_n_overlap_result,
        'groundtruth_ranking': groundtruth_ranking,
        'predicted_ranking': predicted_ranking,
        'predicted_z_hat': predicted_scores,
        'groundtruth_z': groundtruth_labels
    }

def create_visualizations(evaluation_results: Dict, params: np.ndarray, 
                          feature_names: List[str], output_dir: Path):
    """创建可视化图表。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    weights = params[:-1]
    bias = params[-1]
    
    # 1. 权重分布图 (排除偏置)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(weights)), weights, color='salmon', alpha=0.7)
    ax.set_xlabel('Features')
    ax.set_ylabel('Weight')
    ax.set_title(f'Optimized Feature Weights (Bias b: {bias:.4f})')
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * np.sign(height) if height!=0 else 0.01,
                f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_weights_logistic.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 预测概率 vs Groundtruth 标签（散点图）
    common_vars = list(evaluation_results['groundtruth_z'].keys())
    if len(common_vars) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        groundtruth_z = [evaluation_results['groundtruth_z'][var] for var in common_vars]
        predicted_z_hat = [evaluation_results['predicted_z_hat'][var] for var in common_vars]
        
        ax.scatter(groundtruth_z, predicted_z_hat, alpha=0.6, s=50)
        ax.set_xlabel('Groundtruth Label (z: 0 or 1)')
        ax.set_ylabel('Predicted Probability (z_hat)')
        ax.set_title('Predicted Probability vs Groundtruth Label')
        ax.set_xticks([0, 1])
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'probability_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

# --- 主函数 (修改: 调用新函数和指标) ---

def main():
    """主函数。"""
    # 引入 argparse 确保脚本可以运行
    import argparse
    parser = argparse.ArgumentParser(description="Optimize feature weights based on mutation testing groundtruth (Logistic Regression model)")
    parser.add_argument("--groundtruth-file", default=Path("/data/fhj/sva-var/verify/ibex_if_stage/mutation_results_fixed.json"), type=Path,
                        help="Path to mutation_testing_results.json file")
    parser.add_argument("--features-file", default=Path("/data/fhj/sva-var/verify/ibex_if_stage/data/vars/vars.json"), type=Path,
                        help="Path to features JSON file")
    parser.add_argument("--output-dir", type=Path, default=Path("./ibex_if_stage/weight_optimization_logistic"),
                        help="Output directory for results")
    parser.add_argument("--top-n", type=int, default=20,
                        help="The number of top variables to label as 1 (groundtruth z).")
    
    args = parser.parse_args()
    
    if not args.groundtruth_file.exists():
        print(f"Error: Groundtruth file not found: {args.groundtruth_file}")
        return
    if not args.features_file.exists():
        print(f"Error: Features file not found: {args.features_file}")
        return
    
    # 加载数据和创建二元标签 z
    groundtruth_labels, features, feature_names, top_n = load_data(args.groundtruth_file, args.features_file, args.top_n)
    
    if len(features) < 2:
        print(f"Error: Found only {len(features)} matching variables. Need at least 2 for optimization.")
        return
    
    print(f"Using {len(features)} variables for optimization, targeting Top {top_n} classification.")
    
    # 优化参数 (w 和 b)
    optimal_params = optimize_logistic_regression_weights(groundtruth_labels, features, feature_names)
    
    # 评估结果
    evaluation_results = evaluate_results(groundtruth_labels, features, optimal_params, feature_names, top_n)
    
    # 打印结果
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"\nOptimized Parameters:")
    for feature, weight in zip(feature_names, optimal_params[:-1]):
        print(f"  {feature:30}: w = {weight:.4f}")
    print(f"  {'Bias':30}: b = {optimal_params[-1]:.4f}")
    
    print(f"\nEvaluation Classification Metrics:")
    print(f"  BCE Loss (Optimal):  {binary_cross_entropy_loss(optimal_params, groundtruth_labels, features, feature_names):.4f}")
    print(f"  Accuracy:            {evaluation_results['accuracy']:.4f}")
    print(f"  F1-Score:            {evaluation_results['f1_score']:.4f}")
    print(f"  Precision:           {evaluation_results['precision']:.4f}")
    print(f"  Recall:              {evaluation_results['recall']:.4f}")
    print(f"  AUC-ROC:             {evaluation_results['auc_roc']:.4f}")
    print(f"  Top {top_n} Overlap (Ranking by z_hat): {evaluation_results[f'top_{top_n}_overlap']:.2%}")

    
    # 保存结果
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存参数和评估结果
    results = {
        'optimal_weights': dict(zip(feature_names, optimal_params[:-1].tolist())),
        'optimal_bias': optimal_params[-1].item(),
        'evaluation_metrics': {k: v for k, v in evaluation_results.items() if not isinstance(v, (dict, list))},
        'groundtruth_z_ranking': evaluation_results['groundtruth_ranking'],
        'predicted_z_hat_ranking': evaluation_results['predicted_ranking'],
    }
    
    with open(args.output_dir / 'optimization_results_logistic.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 创建可视化
    print("\nCreating visualizations...")
    create_visualizations(evaluation_results, optimal_params, feature_names, args.output_dir / "visualizations")
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    # 需要安装 scikit-learn 来运行评估部分
    try:
        from sklearn.metrics import accuracy_score
    except ImportError:
        print("Required package 'scikit-learn' not found. Please install it with 'pip install scikit-learn'.")
        # 退出或运行不包含 sklearn 的部分
        # 为了保证代码可运行性，这里不做强制退出
        pass 
        
    main()