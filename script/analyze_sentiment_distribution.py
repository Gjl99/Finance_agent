#!/usr/bin/env python3
"""
情感分类数据分布分析脚本
分析 nasdaq_news_sentiment 数据集中情感标签的分布情况
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import glob
import os

# 使用英文避免中文字体问题
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_sentiment_data():
    """加载情感数据集"""
    print("正在加载数据集...")
    dataset_dir = "/home/data1/gjl/more_learning/shock_invest_Agent/Finance/datasets/nasdaq_news_sentiment/benstaf___nasdaq_news_sentiment/default-9e4bd3a263f08264/0.0.0/04532f608f416991dea9b9952775102f8e2216e5"
    arrow_files = glob.glob(os.path.join(dataset_dir, "*.arrow"))
    
    if not arrow_files:
        raise FileNotFoundError(f"未找到 Arrow 文件: {dataset_dir}")
    
    print(f"找到 {len(arrow_files)} 个数据文件")
    
    # 加载数据集
    ds = load_dataset("arrow", data_files=arrow_files)
    df = ds['train'].to_pandas()
    
    print(f"数据集总行数: {len(df):,}")
    return df

def analyze_sentiment_distribution(df):
    """分析情感分布"""
    print("\n" + "="*60)
    print("情感分布统计分析")
    print("="*60)
    
    # 过滤有效数据
    valid_df = df[df['sentiment_deepseek'].notna()]
    print(f"\n有效情感标签数量: {len(valid_df):,} ({len(valid_df)/len(df)*100:.2f}%)")
    print(f"缺失情感标签数量: {len(df) - len(valid_df):,} ({(len(df) - len(valid_df))/len(df)*100:.2f}%)")
    
    # 统计各情感等级的数量
    sentiment_counts = valid_df['sentiment_deepseek'].value_counts().sort_index()
    
    print("\n情感等级分布:")
    print("-" * 60)
    total = len(valid_df)
    
    sentiment_labels = {
        0.0: "未知 (Unknown)",
        1.0: "非常负面 (Very Negative)",
        2.0: "负面 (Negative)",
        3.0: "中性 (Neutral)",
        4.0: "正面 (Positive)",
        5.0: "非常正面 (Very Positive)"
    }
    
    stats_data = []
    for sentiment, count in sentiment_counts.items():
        label = sentiment_labels.get(sentiment, f"未知 ({sentiment})")
        percentage = (count / total) * 100
        print(f"等级 {int(sentiment)} - {label:30s}: {count:7,} ({percentage:6.2f}%)")
        stats_data.append({
            'sentiment': int(sentiment),
            'label': label,
            'count': count,
            'percentage': percentage
        })
    
    print("-" * 60)
    print(f"总计: {total:,}")
    
    return pd.DataFrame(stats_data), valid_df

def create_comparison_charts(stats_df, valid_df):
    """创建原始分布和平衡后分布的对比柱状图"""
    print("\n正在生成对比柱状图...")
    
    # 准备数据 - 过滤掉 Level 0
    original_data = stats_df[stats_df['sentiment'] != 0].copy()
    
    # 创建平衡后的数据(每类最多3000条)
    balanced_data = original_data.copy()
    balanced_data['balanced_count'] = balanced_data['count'].apply(lambda x: min(x, 3000))
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Sentiment Distribution Analysis: Original vs Balanced (Max 3000 per class)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    
    # 图1: 原始分布
    sentiments = original_data['sentiment'].values
    counts = original_data['count'].values
    
    bars1 = ax1.bar(sentiments, counts, color=colors[:len(sentiments)], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Sentiment Level', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax1.set_title('Original Distribution', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xticks(sentiments)
    ax1.set_xticklabels([f"Level {int(s)}" for s in sentiments])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(counts) * 1.15)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 添加3000基准线
    ax1.axhline(y=3000, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Balance Threshold (3000)')
    ax1.legend(loc='upper right', fontsize=10)
    
    # 图2: 平衡后分布 - 纵坐标固定为3500
    balanced_counts = balanced_data['balanced_count'].values
    
    bars2 = ax2.bar(sentiments, balanced_counts, color=colors[:len(sentiments)], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Sentiment Level', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax2.set_title('Balanced Distribution (Max 3000 per class)', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks(sentiments)
    ax2.set_xticklabels([f"Level {int(s)}" for s in sentiments])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 3500)  # 固定Y轴最大值为3500，留出标签空间
    
    # 添加数值标签
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        original_count = counts[i]
        
        # 显示数值
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 如果被截断,显示原始数值
        if height < original_count:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 200,
                    f'(was {int(original_count):,})',
                    ha='center', va='bottom', fontsize=8, color='red', style='italic')
    
    # 添加3000基准线
    ax2.axhline(y=3000, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Balance Threshold')
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = '/home/data1/gjl/more_learning/shock_invest_Agent/Finance/sentiment_comparison_charts.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图表已保存: {output_file}")
    
    # 打印统计信息
    print("\n对比统计:")
    print("-" * 70)
    print(f"{'Level':<8} {'原始数据':<12} {'平衡后数据':<12} {'削减量':<15}")
    print("-" * 70)
    for _, row in original_data.iterrows():
        original = int(row['count'])
        balanced = min(original, 3000)
        reduction = original - balanced
        reduction_pct = (reduction / original * 100) if original > 0 else 0
        print(f"{int(row['sentiment']):<8} {original:<12,} {balanced:<12,} {reduction:>6,} (-{reduction_pct:.1f}%)")
    print("-" * 70)
    print(f"{'总计':<8} {int(original_data['count'].sum()):<12,} {int(balanced_data['balanced_count'].sum()):<12,}")
    
    plt.close()
    return output_file

def main():
    """主函数"""
    print("\n" + "="*70)
    print("情感分类数据分布分析工具")
    print("="*70 + "\n")
    
    try:
        # 1. 加载数据
        df = load_sentiment_data()
        
        # 2. 分析分布
        stats_df, valid_df = analyze_sentiment_distribution(df)
        
        # 3. 创建对比柱状图
        comparison_chart = create_comparison_charts(stats_df, valid_df)
        
        print("\n" + "="*70)
        print("分析完成!")
        print("="*70)
        print(f"\n生成的文件:")
        print(f"  1. 对比柱状图: {comparison_chart}")
        print("\n提示: 使用图片查看器打开 PNG 文件查看可视化结果")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
