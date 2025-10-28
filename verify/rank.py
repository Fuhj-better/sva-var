#!/usr/bin/env python3
"""
突变测试结果分析工具 - 带可视化功能
用于对突变测试结果中的变量进行排序和分析，并生成图表
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def analyze_and_rank_variables(mutation_testing_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析突变测试结果并对变量进行排序。
    """
    variable_results = mutation_testing_results.get("variable_results", {})
    
    # 收集每个变量的统计信息
    variable_stats = {}
    
    for variable_name, result in variable_results.items():
        if "error" in result:
            continue  # 跳过有错误的变量
            
        total_mutations = result.get("total_mutations", 0)
        killed_mutations = result.get("killed_mutations", 0)
        mutation_score = result.get("mutation_score", 0)
        assertion_count = result.get("assertion_count", 0)
        
        # 突变类型统计
        mutation_type_stats = result.get("mutation_type_statistics", {})
        
        # 计算跳过的突变数量
        skipped_mutations = sum(
            stats.get("skipped", 0) for stats in mutation_type_stats.values()
        )
        
        # 计算有效的突变测试数量（非跳过的）
        effective_mutations = total_mutations - skipped_mutations
        
        # 计算突变类型覆盖度（有多少种突变类型被有效测试）
        mutation_type_coverage = len([
            stats for stats in mutation_type_stats.values() 
            if stats.get("total", 0) > stats.get("skipped", 0)
        ])
        
        # 计算综合评分
        comp_score = (
            mutation_score * 0.4 +  # 突变分数权重 40%
            (killed_mutations / max(effective_mutations, 1)) * 0.3 +  # 有效杀死率权重 30%
            (min(assertion_count / 20, 1.0)) * 0.2 +  # 断言数量权重 20%（最多20个断言）
            (mutation_type_coverage / max(len(mutation_type_stats), 1)) * 0.1  # 多样性权重 10%
        )
        
        stats = {
            "variable_name": variable_name,
            "assertion_count": assertion_count,
            "total_mutations": total_mutations,
            "killed_mutations": killed_mutations,
            "mutation_score": mutation_score,
            "effective_mutations": effective_mutations,
            "skipped_mutations": skipped_mutations,
            "mutation_type_count": len(mutation_type_stats),
            "mutation_type_coverage": mutation_type_coverage,
            "comprehensive_score": comp_score,
            "mutation_type_details": mutation_type_stats
        }
        
        variable_stats[variable_name] = stats
    
    # 定义不同的排序策略
    ranking_strategies = {
        "by_mutation_score": sorted(variable_stats.values(), key=lambda x: x["mutation_score"], reverse=True),
        "by_killed_count": sorted(variable_stats.values(), key=lambda x: x["killed_mutations"], reverse=True),
        "by_assertion_count": sorted(variable_stats.values(), key=lambda x: x["assertion_count"], reverse=True),
        "by_mutation_diversity": sorted(variable_stats.values(), key=lambda x: (x["mutation_type_count"], x["mutation_type_coverage"]), reverse=True),
        "by_effective_mutations": sorted(variable_stats.values(), key=lambda x: x["effective_mutations"], reverse=True),
        "by_comprehensive_score": sorted(variable_stats.values(), key=lambda x: x["comprehensive_score"], reverse=True)
    }
    
    # 生成最终报告
    analysis_report = {
        "summary": {
            "total_variables": len(variable_stats),
            "average_mutation_score": sum(stats["mutation_score"] for stats in variable_stats.values()) / len(variable_stats) if variable_stats else 0,
            "average_comprehensive_score": sum(stats["comprehensive_score"] for stats in variable_stats.values()) / len(variable_stats) if variable_stats else 0,
            "total_mutations_tested": mutation_testing_results.get("total_mutations_tested", 0),
            "total_killed_mutations": mutation_testing_results.get("total_killed_mutations", 0),
            "overall_mutation_score": mutation_testing_results.get("overall_mutation_score", 0)
        },
        "rankings": ranking_strategies,
        "variable_details": variable_stats
    }
    
    return analysis_report

def create_visualizations(analysis_report: Dict[str, Any], output_dir: Path, top_n: int = 20):
    """
    创建各种可视化图表。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rankings = analysis_report["rankings"]
    variable_details = analysis_report["variable_details"]
    
    # 1. 突变分数排名图 - 修复：变量名作为x轴
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 突变分数柱状图 - 修复：变量名在x轴
    top_vars = rankings["by_mutation_score"][:top_n]
    variables = [var["variable_name"] for var in top_vars]
    scores = [var["mutation_score"] for var in top_vars]
    
    bars = ax1.bar(range(len(variables)), scores, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Variables')
    ax1.set_ylabel('Mutation Score')
    ax1.set_title(f'Top {top_n} Variables by Mutation Score')
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # 设置x轴标签
    ax1.set_xticks(range(len(variables)))
    ax1.set_xticklabels(variables, rotation=45, ha='right')
    
    # 在柱子上添加数值标签
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.1%}', ha='center', va='bottom', fontsize=8)
    
    # 综合评分柱状图 - 修复：变量名在x轴
    top_vars_comp = rankings["by_comprehensive_score"][:top_n]
    variables_comp = [var["variable_name"] for var in top_vars_comp]
    scores_comp = [var["comprehensive_score"] for var in top_vars_comp]
    
    bars_comp = ax2.bar(range(len(variables_comp)), scores_comp, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Variables')
    ax2.set_ylabel('Comprehensive Score')
    ax2.set_title(f'Top {top_n} Variables by Comprehensive Score')
    ax2.set_ylim(0, 1)
    
    # 设置x轴标签
    ax2.set_xticks(range(len(variables_comp)))
    ax2.set_xticklabels(variables_comp, rotation=45, ha='right')
    
    # 在柱子上添加数值标签
    for bar, score in zip(bars_comp, scores_comp):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mutation_score_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 水平柱状图版本（如果您还是喜欢水平的话）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # 突变分数水平柱状图
    top_vars = rankings["by_mutation_score"][:top_n]
    variables = [var["variable_name"] for var in top_vars]
    scores = [var["mutation_score"] for var in top_vars]
    
    bars = ax1.barh(range(len(variables)), scores, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Mutation Score')
    ax1.set_ylabel('Variables')
    ax1.set_title(f'Top {top_n} Variables by Mutation Score (Horizontal)')
    ax1.set_xlim(0, 1)
    ax1.xaxis.set_major_formatter(PercentFormatter(1.0))
    
    # 设置y轴标签
    ax1.set_yticks(range(len(variables)))
    ax1.set_yticklabels(variables)
    
    # 在柱子上添加数值标签
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.1%}', ha='left', va='center', fontsize=8)
    
    # 综合评分水平柱状图
    top_vars_comp = rankings["by_comprehensive_score"][:top_n]
    variables_comp = [var["variable_name"] for var in top_vars_comp]
    scores_comp = [var["comprehensive_score"] for var in top_vars_comp]
    
    bars_comp = ax2.barh(range(len(variables_comp)), scores_comp, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Comprehensive Score')
    ax2.set_ylabel('Variables')
    ax2.set_title(f'Top {top_n} Variables by Comprehensive Score (Horizontal)')
    ax2.set_xlim(0, 1)
    
    # 设置y轴标签
    ax2.set_yticks(range(len(variables_comp)))
    ax2.set_yticklabels(variables_comp)
    
    # 在柱子上添加数值标签
    for bar, score in zip(bars_comp, scores_comp):
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mutation_score_ranking_horizontal.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 杀死数量 vs 突变分数散点图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    all_variables = list(variable_details.values())
    killed_counts = [var["killed_mutations"] for var in all_variables]
    mutation_scores = [var["mutation_score"] for var in all_variables]
    assertion_counts = [var["assertion_count"] for var in all_variables]
    variable_names = [var["variable_name"] for var in all_variables]
    
    # 使用断言数量作为点的大小
    sizes = [min(count * 10 + 20, 200) for count in assertion_counts]
    
    scatter = ax.scatter(killed_counts, mutation_scores, s=sizes, alpha=0.6, 
                        c=mutation_scores, cmap='viridis')
    
    ax.set_xlabel('Killed Mutations')
    ax.set_ylabel('Mutation Score')
    ax.set_title('Mutation Score vs Killed Mutations (Bubble size = Assertion Count)')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Mutation Score')
    
    # 为一些重要的点添加标签
    for i, (x, y, name) in enumerate(zip(killed_counts, mutation_scores, variable_names)):
        if x > np.percentile(killed_counts, 75) or y > np.percentile(mutation_scores, 75):
            ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mutation_vs_killed_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 断言数量分布图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 断言数量直方图
    assertion_counts_data = [var["assertion_count"] for var in all_variables]
    ax1.hist(assertion_counts_data, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax1.set_xlabel('Assertion Count')
    ax1.set_ylabel('Number of Variables')
    ax1.set_title('Distribution of Assertion Counts per Variable')
    ax1.grid(True, alpha=0.3)
    
    # 突变分数分布图
    mutation_scores_data = [var["mutation_score"] for var in all_variables]
    ax2.hist(mutation_scores_data, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    ax2.set_xlabel('Mutation Score')
    ax2.set_ylabel('Number of Variables')
    ax2.set_title('Distribution of Mutation Scores')
    ax2.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 突变类型多样性图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    type_counts = [var["mutation_type_count"] for var in all_variables]
    coverage_rates = [var["mutation_type_coverage"] / max(var["mutation_type_count"], 1) 
                     for var in all_variables]
    
    scatter = ax.scatter(type_counts, coverage_rates, s=50, alpha=0.6, 
                        c=mutation_scores, cmap='plasma')
    
    ax.set_xlabel('Mutation Type Count')
    ax.set_ylabel('Mutation Type Coverage Rate')
    ax.set_title('Mutation Type Diversity vs Coverage Rate')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Mutation Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mutation_type_diversity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 综合评分雷达图（前6个变量）
    if len(all_variables) >= 6:
        create_radar_chart(rankings["by_comprehensive_score"][:6], output_dir)
    
    print(f"Visualizations saved to: {output_dir}")
def create_radar_chart(top_variables: List[Dict], output_dir: Path):
    """
    创建雷达图显示前几个变量的综合表现。
    """
    # 选择要显示的指标
    categories = ['Mutation Score', 'Killed Rate', 'Assertion Coverage', 'Type Diversity']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 为每个变量计算雷达图数据
    for i, var in enumerate(top_variables[:6]):
        # 标准化各个指标到0-1范围
        mutation_score = var["mutation_score"]
        killed_rate = var["killed_mutations"] / max(var["effective_mutations"], 1)
        assertion_coverage = min(var["assertion_count"] / 20, 1.0)  # 最多20个断言
        type_diversity = var["mutation_type_coverage"] / max(var["mutation_type_count"], 1)
        
        values = [mutation_score, killed_rate, assertion_coverage, type_diversity]
        values += values[:1]  # 闭合雷达图
        
        # 计算角度
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        # 绘制雷达图
        ax.plot(angles, values, 'o-', linewidth=2, label=var["variable_name"], alpha=0.7)
        ax.fill(angles, values, alpha=0.1)
    
    # 设置类别标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Top Variables Performance Radar Chart', size=14, y=1.08)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
def print_ranking_report(analysis_report: Dict[str, Any], top_n: int = 20):
    """
    打印排序报告。
    """
    rankings = analysis_report["rankings"]
    summary = analysis_report["summary"]
    
    print("\n" + "="*80)
    print("MUTATION TESTING VARIABLE RANKING REPORT")
    print("="*80)
    
    print(f"\nSUMMARY:")
    print(f"  Total Variables: {summary['total_variables']}")
    print(f"  Average Mutation Score: {summary['average_mutation_score']:.2%}")
    print(f"  Average Comprehensive Score: {summary['average_comprehensive_score']:.3f}")
    print(f"  Overall Mutation Score: {summary['overall_mutation_score']:.2%}")
    print(f"  Total Mutations Tested: {summary['total_mutations_tested']}")
    print(f"  Total Killed Mutations: {summary['total_killed_mutations']}")
    
    # 按突变分数排序 - 使用 top_n 参数
    print(f"\n{'='*70}")
    print(f"RANKING BY MUTATION SCORE (Primary Metric) - Top {top_n}")
    print(f"{'='*70}")
    print(f"{'Rank':<4} {'Variable':<20} {'Score':<8} {'Killed':<8} {'Total':<8} {'Assertions':<10} {'Types':<6}")
    print(f"{'-'*70}")
    for i, var in enumerate(rankings["by_mutation_score"][:top_n], 1):
        print(f"{i:<4} {var['variable_name']:<20} {var['mutation_score']:>7.1%} "
              f"{var['killed_mutations']:>7} {var['total_mutations']:>7} "
              f"{var['assertion_count']:>9} {var['mutation_type_count']:>5}")
    
    # 按综合评分排序 - 使用 top_n 参数
    print(f"\n{'='*70}")
    print(f"RANKING BY COMPREHENSIVE SCORE (Recommended) - Top {top_n}")
    print(f"{'='*70}")
    print(f"{'Rank':<4} {'Variable':<20} {'CompScore':<10} {'Mutation':<8} {'Killed':<8} {'Assertions':<10}")
    print(f"{'-'*70}")
    for i, var in enumerate(rankings["by_comprehensive_score"][:top_n], 1):
        print(f"{i:<4} {var['variable_name']:<20} {var['comprehensive_score']:>9.3f} "
              f"{var['mutation_score']:>7.1%} {var['killed_mutations']:>7} {var['assertion_count']:>9}")
    
    # 按杀死数量排序 - 使用 top_n 参数
    print(f"\n{'='*60}")
    print(f"RANKING BY KILLED MUTATIONS - Top {top_n}")
    print(f"{'='*60}")
    print(f"{'Rank':<4} {'Variable':<20} {'Killed':<8} {'Score':<8} {'Effective':<10}")
    print(f"{'-'*60}")
    for i, var in enumerate(rankings["by_killed_count"][:top_n], 1):
        print(f"{i:<4} {var['variable_name']:<20} {var['killed_mutations']:>7} "
              f"{var['mutation_score']:>7.1%} {var['effective_mutations']:>9}")
    
    # 按断言数量排序 - 使用 top_n 参数
    print(f"\n{'='*60}")
    print(f"RANKING BY ASSERTION COUNT - Top {top_n}")
    print(f"{'='*60}")
    print(f"{'Rank':<4} {'Variable':<20} {'Assertions':<10} {'Score':<8} {'Killed':<8}")
    print(f"{'-'*60}")
    for i, var in enumerate(rankings["by_assertion_count"][:top_n], 1):
        print(f"{i:<4} {var['variable_name']:<20} {var['assertion_count']:>9} "
              f"{var['mutation_score']:>7.1%} {var['killed_mutations']:>7}")
    
    # 按突变类型多样性排序 - 使用 top_n 参数
    print(f"\n{'='*60}")
    print(f"RANKING BY MUTATION DIVERSITY - Top {top_n}")
    print(f"{'='*60}")
    print(f"{'Rank':<4} {'Variable':<20} {'Types':<6} {'Coverage':<9} {'Score':<8}")
    print(f"{'-'*60}")
    for i, var in enumerate(rankings["by_mutation_diversity"][:top_n], 1):
        coverage_rate = var['mutation_type_coverage'] / max(var['mutation_type_count'], 1)
        print(f"{i:<4} {var['variable_name']:<20} {var['mutation_type_count']:>5} "
              f"{coverage_rate:>8.1%} {var['mutation_score']:>7.1%}")

def print_detailed_analysis(analysis_report: Dict[str, Any], variable_name: str = None, top_n: int = 20):
    """
    打印详细分析。
    """
    variable_details = analysis_report["variable_details"]
    
    if variable_name and variable_name in variable_details:
        # 打印单个变量的详细分析
        var = variable_details[variable_name]
        print(f"\n{'='*80}")
        print(f"DETAILED ANALYSIS: {variable_name}")
        print(f"{'='*80}")
        print(f"Assertion Count: {var['assertion_count']}")
        print(f"Mutation Score: {var['mutation_score']:.2%}")
        print(f"Killed Mutations: {var['killed_mutations']} / {var['total_mutations']}")
        print(f"Effective Mutations: {var['effective_mutations']}")
        print(f"Skipped Mutations: {var['skipped_mutations']}")
        print(f"Mutation Types: {var['mutation_type_count']}")
        print(f"Mutation Type Coverage: {var['mutation_type_coverage']}")
        print(f"Comprehensive Score: {var['comprehensive_score']:.3f}")
        
        print(f"\nMutation Type Details:")
        for mutation_type, stats in var['mutation_type_details'].items():
            tested = stats['total'] - stats['skipped']
            killed_rate = stats['killed'] / tested if tested > 0 else 0
            print(f"  {mutation_type:15} | Total: {stats['total']:3d} | Tested: {tested:3d} | "
                  f"Killed: {stats['killed']:3d} | Rate: {killed_rate:6.1%}")
    
    else:
        # 打印所有变量的简要统计 - 使用 top_n 参数
        print(f"\n{'='*80}")
        print(f"VARIABLE STATISTICS SUMMARY - Top {top_n}")
        print(f"{'='*80}")
        print(f"{'Variable':<20} {'Score':<8} {'Killed':<8} {'Assertions':<10} {'Types':<6} {'CompScore':<10}")
        print(f"{'-'*80}")
        
        # 按综合评分排序并取前 top_n 个
        sorted_vars = sorted(variable_details.values(), key=lambda x: x["comprehensive_score"], reverse=True)
        for var in sorted_vars[:top_n]:
            print(f"{var['variable_name']:<20} {var['mutation_score']:>7.1%} {var['killed_mutations']:>7} "
                  f"{var['assertion_count']:>9} {var['mutation_type_count']:>5} {var['comprehensive_score']:>9.3f}")

def save_ranking_report(analysis_report: Dict[str, Any], output_dir: Path, top_n: int = 20):
    """
    保存排序报告到文件。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存JSON报告
    report_file = output_dir / "variable_ranking_analysis.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)
    
    # 保存可读的文本报告 - 使用 top_n 参数
    text_report_file = output_dir / "variable_ranking_report.txt"
    with open(text_report_file, 'w', encoding='utf-8') as f:
        # 重定向输出到文件
        original_stdout = sys.stdout
        sys.stdout = f
        print_ranking_report(analysis_report, top_n)
        print_detailed_analysis(analysis_report, top_n=top_n)
        sys.stdout = original_stdout
    
    print(f"\nRanking report saved to:")
    print(f"  JSON: {report_file}")
    print(f"  Text: {text_report_file} (showing top {top_n} variables)")

def main():
    """
    主函数。
    """
    parser = argparse.ArgumentParser(description="Analyze mutation testing results with visualizations")
    parser.add_argument("--results-file", default=Path("/data/fhj/sva-var/verify/ibex_if_stage/mutation_results_fixed.json"), type=Path, 
                       help="Path to mutation_testing_results.json file")
    parser.add_argument("--output-dir", type=Path, default=Path("./ibex_if_stage/analysis_results"),
                       help="Output directory for analysis reports and visualizations")
    parser.add_argument("--top-n", type=int, default=100,
                       help="Number of top variables to show in rankings and visualizations")
    parser.add_argument("--no-viz", action="store_true",
                       help="Skip visualization generation")
    
    args = parser.parse_args()
    
    # 检查结果文件是否存在
    if not args.results_file.exists():
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)
    
    # 加载突变测试结果
    print(f"Loading mutation testing results from: {args.results_file}")
    with open(args.results_file, 'r', encoding='utf-8') as f:
        mutation_results = json.load(f)
    
    # 分析和排序变量
    print("Analyzing and ranking variables...")
    analysis_report = analyze_and_rank_variables(mutation_results)
    
    # 打印报告 - 传递 top_n 参数
    print_ranking_report(analysis_report, args.top_n)
    
    # 保存报告 - 传递 top_n 参数
    save_ranking_report(analysis_report, args.output_dir, args.top_n)
    
    # 生成可视化图表 - 传递 top_n 参数
    if not args.no_viz:
        print("Generating visualizations...")
        create_visualizations(analysis_report, args.output_dir / "visualizations", args.top_n)
    else:
        print("Skipping visualization generation")

if __name__ == "__main__":
    main()