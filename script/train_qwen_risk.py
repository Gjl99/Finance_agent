import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import warnings
import sys
# 移除了 argparse

warnings.filterwarnings("ignore")

# --- 1. 在此处定义所有参数 ---
# 指定使用哪两张显卡 (例如: "0,1" 表示使用第0和第1张显卡)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 修改这里选择你想使用的显卡编号

MODEL_NAME = "./model_train/Qwen3-4B"  # 使用已缓存的 Qwen3-4B 模型
DATASET_NAME = "benstaf/risk_nasdaq"
DATA_FILE = "risk_deepseek_cleaned_nasdaq_news_full.csv"  # 只使用 deepseek 文件
CACHE_DIR = "/home/data1/gjl/more_learning/shock_invest_Agent/Finance/datasets/risk_nasdaq"
OUTPUT_DIR = "./model_train/qwen_risk" # 训练后模型的保存目录
MAX_LENGTH = 512 # 分词最大长度

# --- 2. 加载数据 (从指定缓存) ---
def load_and_preprocess_data(dataset_name, data_file, cache_dir=None):
    """
    从本地缓存加载指定的数据文件。
    """
    print(f"正在从本地缓存加载数据集: {dataset_name}")
    print(f"数据文件: {data_file}")
    print(f"缓存目录: {cache_dir}")
    
    try:
        dataset = load_dataset(
            dataset_name, 
            data_files=data_file,
            split="train", 
            cache_dir=cache_dir
        )
    except Exception as e:
        print(f"从缓存 {cache_dir} 加载数据集失败: {e}", file=sys.stderr)
        print("请确保已先运行 download_dataset.py", file=sys.stderr)
        sys.exit(1)
        
    df = dataset.to_pandas()
    
    print(f"原始数据数量: {len(df)}")
    print(f"数据集列名: {df.columns.tolist()}")
    
    # 过滤有效数据
    df = df[df['Lsa_summary'].notna() & df['risk_deepseek'].notna()]
    df = df[df['risk_deepseek'] != 0]
    
    print(f"有效数据数量: {len(df)}")
    print(f"风险分布: \n{df['risk_deepseek'].value_counts().sort_index()}")
    
    return df

def create_prompt_template(text, risk_score, stock_symbol="STOCK"):
    """创建训练提示模板"""
    system_prompt = "Forget all your previous instructions. You are a financial expert specializing in risk assessment for stock recommendations. Based on a specific stock, provide a risk score from 1 to 5, where: 1 indicates very low risk, 2 indicates low risk, 3 indicates moderate risk (default if the news lacks any clear indication of risk), 4 indicates high risk, and 5 indicates very high risk. 1 summarized news will be passed in each time. Provide the score in the format shown below in the response from the assistant."
    user_content = f"News to Stock Symbol -- {stock_symbol}: {text}"
    
    conversation = f"""System: {system_prompt}

User: News to Stock Symbol -- AAPL: Apple (AAPL) increases 22%
Assistant: 3

User: News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30%
Assistant: 4

User: News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15
Assistant: 3

User: {user_content}
Assistant: {risk_score}"""
    
    return conversation

# --- 3. 处理和分词数据 ---
def prepare_dataset(df, tokenizer, max_length=MAX_LENGTH):
    """准备训练数据集 (在内存中处理)"""
    print("正在准备和分词数据集...")
    
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        text = row['Lsa_summary']
        risk_score = int(row['risk_deepseek'])
        stock_symbol = row.get('Stock_symbol', 'STOCK')
        
        if pd.isna(text) or text == '':
            continue
            
        prompt = create_prompt_template(text, risk_score, stock_symbol)
        texts.append(prompt)
        labels.append(risk_score)
    
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(eval_texts)}")
    
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    eval_dataset = Dataset.from_dict({'text': eval_texts, 'label': eval_labels})
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized
    
    # 在内存中进行分词 (num_proc可以加速)
    train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names, num_proc=4)
    eval_tokenized = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names, num_proc=4)
    
    return train_tokenized, eval_tokenized

# --- 4. 创建模型 ---
def create_model_and_tokenizer(model_name, cache_dir=None):
    """创建模型和分词器"""
    print(f"正在加载模型和分词器: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 检测是否在分布式训练环境
    is_distributed = os.environ.get('WORLD_SIZE') is not None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None if is_distributed else "auto",  # DDP训练时不使用device_map
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    # 启用梯度检查点
    model.gradient_checkpointing_enable()
    
    # 准备模型用于训练（不使用kbit，因为我们用的是fp16）
    model.enable_input_require_grads()
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

# --- 5. 训练模型 ---
def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir):
    """训练模型"""
    print("开始训练模型...")
    
    # 检查环境变量，确认torchrun是否正确设置
    if os.environ.get('WORLD_SIZE'):
        print(f"检测到分布式训练: WORLD_SIZE={os.environ['WORLD_SIZE']}, RANK={os.environ['RANK']}, LOCAL_RANK={os.environ['LOCAL_RANK']}")
    else:
        print("未检测到分布式环境，将以单进程模式运行。")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,  # 降低批次大小避免OOM
        gradient_accumulation_steps=16,  # 增加梯度累积保持有效批次大小
        warmup_steps=100,
        learning_rate=1e-4,
        fp16=True, 
        logging_steps=20,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        # gradient_checkpointing 已在模型中启用，无需在这里设置
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"模型已保存到: {output_dir}")

def main():
    
    # 1. 加载数据 (使用顶部定义的全局变量)
    df = load_and_preprocess_data(DATASET_NAME, DATA_FILE, CACHE_DIR)
    
    # 2. 创建模型和分词器 (使用默认缓存)
    model, tokenizer = create_model_and_tokenizer(MODEL_NAME, cache_dir=None)
    
    # 3. 准备数据集 (在内存中处理)
    train_dataset, eval_dataset = prepare_dataset(df, tokenizer)
    
    # 4. 训练模型 (使用顶部定义的全局变量)
    train_model(model, tokenizer, train_dataset, eval_dataset, OUTPUT_DIR)
    
    print("训练完成！")

if __name__ == "__main__":
    main()