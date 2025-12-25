#!/usr/bin/env python3
"""
下载 benstaf/risk_nasdaq 数据集中的 risk_deepseek_cleaned_nasdaq_news_full.csv 文件到本地目录
"""
import os
import sys
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")

def download_dataset():
    """
    下载 risk_deepseek_cleaned_nasdaq_news_full.csv 文件到指定目录
    """
    dataset_name = "benstaf/nasdaq_news_sentiment"
    data_file = "sentiment_deepseek_new_cleaned_nasdaq_news_full.csv"
    local_dir = "/home/data1/gjl/more_learning/shock_invest_Agent/Finance/datasets/nasdaq_news_sentiment"
    
    print("="*60)
    print("数据集下载工具 - benstaf/nasdaq_news_sentiment")
    print("="*60)
    print(f"\n数据集: {dataset_name}")
    print(f"文件: {data_file}")
    print(f"保存到: {local_dir}\n")
    
    try:
        # 创建本地目录
        os.makedirs(local_dir, exist_ok=True)
        
        # 使用 load_dataset 加载指定的 CSV 文件
        print("正在下载...")
        dataset = load_dataset(
            dataset_name,
            data_files=data_file,
            cache_dir=local_dir,
            split="train"
        )
        
        print("\n" + "="*60)
        print("✓ 下载成功!")
        print(f"数据集行数: {len(dataset)}")
        print(f"数据集列名: {dataset.column_names}")
        print(f"缓存位置: {local_dir}")
        print("="*60)
        
        # 显示前几行数据
        print("\n数据预览 (前3行):")
        df = dataset.to_pandas()
        print(df.head(3))
        
        return dataset
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}", file=sys.stderr)
        print("\n请检查:")
        print("1. 网络连接是否正常")
        print("2. Hugging Face 是否可访问")
        print("3. 数据集名称和文件名是否正确")
        sys.exit(1)

def main():
    download_dataset()
    print("\n下载完成!")

if __name__ == "__main__":
    main()