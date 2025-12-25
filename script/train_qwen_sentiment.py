import os
import sys
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5" 
os.environ["SWANLAB_API_KEY"] = "api_key"
SWANLAB_MODE = "cloud"

import torch
import pandas as pd
import numpy as np
import re
import random
import datetime
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
import swanlab

# 过滤警告
warnings.filterwarnings("ignore")

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
PROJECT_ROOT = "/home/data1/gjl/more_learning/shock_invest_Agent/Finance/qwen_sentiment_project"

# [优化] 所有输出文件统一保存在一个实验目录下
EXPERIMENT_NAME = f"qwen_sentiment_{TIMESTAMP}"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, EXPERIMENT_NAME)

# 子目录结构
OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, "checkpoints")  # 模型检查点
LOG_DIR = os.path.join(EXPERIMENT_DIR, "logs")            # Tensorboard/Transformers 日志
SWANLAB_LOG_DIR = os.path.join(EXPERIMENT_DIR, "swanlab") # Swanlab 本地日志
CACHE_DIR = os.path.join(PROJECT_ROOT, "datasets")        # 数据缓存

# 模型配置
MODEL_NAME = "/home/data1/gjl/more_learning/shock_invest_Agent/Finance/model_train/Qwen3-4B" 
DATASET_NAME = "benstaf/nasdaq_news_sentiment"
DATA_FILE = "sentiment_deepseek_new_cleaned_nasdaq_news_full.csv"

MAX_LENGTH = 512 
MAX_SAMPLES_PER_CLASS = 3000

BATCH_SIZE = 8  
GRAD_ACCUM_STEPS = 4 # 有效 Batch = 8 * 4 * GPU数 = 64
LEARNING_RATE = 5e-5  # 降低学习率避免梯度爆炸 
# ====================================================

def balance_dataframe(df, target_col='sentiment_deepseek', max_per_class=3000):
    """数据平衡"""
    df = df[(df[target_col] != 0) & (df[target_col].notna())]
    balanced_dfs = []
    print(f"正在进行数据平衡 (上限: {max_per_class} 条/类)...")
    for sentiment in [1, 2, 3, 4, 5]:
        class_df = df[df[target_col] == sentiment]
        count = len(class_df)
        if count == 0: continue
            
        if count > max_per_class:
            sampled_df = class_df.sample(n=max_per_class, random_state=42)
            balanced_dfs.append(sampled_df)
            print(f"  类别 {sentiment}: 原始 {count} -> 采样后 {max_per_class}")
        else:
            balanced_dfs.append(class_df)
            print(f"  类别 {sentiment}: 原始 {count} -> 保留全部")
            
    final_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    return final_df

def load_and_preprocess_data(dataset_name, data_file, cache_dir=None):
    """加载数据"""
    print(f"正在加载数据集: {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, data_files=data_file, split="train", cache_dir=cache_dir)
    except Exception as e:
        print(f"加载失败: {e}", file=sys.stderr)
        sys.exit(1)
        
    df = dataset.to_pandas()
    df = df[df['Lsa_summary'].notna() & df['sentiment_deepseek'].notna()]
    df = balance_dataframe(df, max_per_class=MAX_SAMPLES_PER_CLASS)
    print(f"最终训练数据量: {len(df)}")
    return df

def prepare_dataset(df, tokenizer, max_length=MAX_LENGTH):
    """数据预处理与 Masking"""
    print("正在构建对话格式并应用标签掩码 (Label Masking)...")
    
    data_list = []
    
    # System Prompt + Few-Shot Examples - 使用更明确的输出格式
    base_history = [
        {"role": "system", "content": "You are a financial sentiment analyzer. Rate news from 1-5 and explain briefly."},
        {"role": "user", "content": "AAPL: Stock price surged 20%"},
        {"role": "assistant", "content": "Score: 5"},
        {"role": "user", "content": "AAPL: Stock price dropped 25%"},
        {"role": "assistant", "content": "Score: 1"},
        {"role": "user", "content": "AAPL: Company announced new product"},
        {"role": "assistant", "content": "Score: 3"}
    ]

    for idx, row in df.iterrows():
        text = row['Lsa_summary']
        try:
            sentiment = int(row['sentiment_deepseek'])
            if sentiment not in [1, 2, 3, 4, 5]: continue
        except: continue
        
        stock_symbol = row.get('Stock_symbol', 'STOCK')
        
        # 深度复制以防污染
        messages = [x.copy() for x in base_history]
        messages.append({"role": "user", "content": f"{stock_symbol}: {text}"})
        
        data_list.append({
            "messages": messages,
            "target": f"Score: {sentiment}"  # 使用更明确的格式
        })

    # 划分数据集
    random.seed(42)
    random.shuffle(data_list)
    split_idx = int(len(data_list) * 0.9)
    train_data = data_list[:split_idx]
    eval_data = data_list[split_idx:]
    
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    def tokenize_function(examples):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        debug_count = 0
        
        for messages, target in zip(examples['messages'], examples['target']):
            # 1. 生成 Prompt
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # 2. 生成 Full Text
            full_text = prompt_text + target + tokenizer.eos_token
            
            # 调试：打印前几个样本
            if debug_count < 2:
                print(f"\n[Tokenize Debug {debug_count}]")
                print(f"Target: '{target}'")
                print(f"Prompt ends: ...{prompt_text[-50:]}")
                print(f"Full text ends: ...{full_text[-80:]}")
                debug_count += 1
            
            # 3. 分别编码
            tokenized_full = tokenizer(
                full_text, 
                truncation=True, 
                max_length=max_length, 
                padding=False, 
                add_special_tokens=False
            )
            
            tokenized_prompt = tokenizer(
                prompt_text, 
                truncation=True, 
                max_length=max_length, 
                padding=False, 
                add_special_tokens=False
            )
            
            input_ids = tokenized_full['input_ids']
            attention_mask = tokenized_full['attention_mask']
            prompt_len = len(tokenized_prompt['input_ids'])
            
            # 4. Masking
            labels = list(input_ids)
            for i in range(prompt_len):
                if i < len(labels):
                    labels[i] = -100 # -100 表示不计算 Loss
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
            
        return {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'labels': labels_list
        }

    print("开始 Tokenize 数据集...")
    # 增加 num_proc 并行处理，加快速度
    train_tokenized = train_dataset.map(tokenize_function, batched=True, batch_size=2000, remove_columns=['messages', 'target'], num_proc=8)
    eval_tokenized = eval_dataset.map(tokenize_function, batched=True, batch_size=2000, remove_columns=['messages', 'target'], num_proc=4)
    
    return train_tokenized, eval_tokenized

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_pred, tokenizer_instance):
    predictions, labels = eval_pred
    pred_sentiments = []
    true_sentiments = []
    debug_printed = 0
    
    # 将 get_score 定义移到前面
    def get_score(text):
        # 先尝试匹配 "Score: X" 格式
        match = re.search(r'Score:\s*([1-5])', text)
        if match:
            return int(match.group(1))
        # 备用：直接查找数字
        nums = re.findall(r'[1-5]', text)
        if nums: return int(nums[-1])
        return 3
    
    for i in range(len(labels)):
        valid_indices = np.where(labels[i] != -100)[0]
        if len(valid_indices) == 0: continue
            
        true_text = tokenizer_instance.decode(labels[i][valid_indices], skip_special_tokens=True).strip()
        pred_text = tokenizer_instance.decode(predictions[i][valid_indices], skip_special_tokens=True).strip()
        
        # 调试输出
        if debug_printed < 3:
            print(f"\n[Eval Debug {debug_printed}]")
            print(f"  True text: '{true_text}'")
            print(f"  Pred text: '{pred_text}'")
            print(f"  True score: {get_score(true_text)}, Pred score: {get_score(pred_text)}")
            debug_printed += 1
            
        true_sentiments.append(get_score(true_text))
        pred_sentiments.append(get_score(pred_text))

    accuracy = accuracy_score(true_sentiments, pred_sentiments)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_sentiments, pred_sentiments, average='weighted', zero_division=0, labels=[1, 2, 3, 4, 5]
    )
    
    class_metrics = {}
    for cls in [1, 2, 3, 4, 5]:
        indices = [k for k, x in enumerate(true_sentiments) if x == cls]
        if indices:
            cls_preds = [pred_sentiments[k] for k in indices]
            cls_trues = [true_sentiments[k] for k in indices]
            class_metrics[f"class_{cls}_acc"] = accuracy_score(cls_trues, cls_preds)
    
    print(f"\nEval Result - Acc: {accuracy:.2%}, F1: {f1:.4f}")
    
    metrics = {
        'eval_accuracy': accuracy,
        'eval_f1': f1,
        'eval_precision': precision,
        'eval_recall': recall
    }
    metrics.update(class_metrics)
    return metrics

def create_model_and_tokenizer(model_name, cache_dir=None):
    print(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 
    
    is_distributed = os.environ.get('WORLD_SIZE') is not None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16, 
        device_map=None if is_distributed else "auto",
        trust_remote_code=True,
        cache_dir=cache_dir
        # 移除 flash_attention_2 以避免兼容性问题
    )
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # 降低rank以提高稳定性
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 减少目标模块
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, eval_dataset):
    is_distributed = os.environ.get('WORLD_SIZE') is not None
    is_main_process = not is_distributed or int(os.environ.get('RANK', 0)) == 0
    
    if is_main_process:
        # [优化] 确保所有日志目录存在
        os.makedirs(SWANLAB_LOG_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        swanlab.init(
            project="qwen-sentiment-training",
            experiment_name=EXPERIMENT_NAME, # 使用时间戳命名
            logdir=SWANLAB_LOG_DIR, # 指定本地日志保存位置
            config={
                "model": MODEL_NAME, 
                "lr": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM_STEPS,
                "lora_rank": 64
            }
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,           # 模型检查点保存位置
        logging_dir=LOG_DIR,             # Tensorboard 日志保存位置
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE, 
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,  # 添加warmup
        max_grad_norm=1.0,  # 梯度裁剪
        fp16=True, 
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        dataloader_num_workers=8,        # 增加 worker 数加快数据加载
        report_to=["swanlab", "tensorboard"], # 同时记录 Swanlab 和 Tensorboard
        gradient_checkpointing=True,
        group_by_length=True,
        ddp_find_unused_parameters=False,
    )
    
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_wrapper,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    print(f"\n{'='*20} 开始训练 {'='*20}")
    print(f"实验目录: {EXPERIMENT_DIR}")
    print(f"Batch Size: {BATCH_SIZE} | Accum Steps: {GRAD_ACCUM_STEPS}")
    
    trainer.train()
    
    # 保存最终模型
    final_save_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"训练完成，最终模型已保存至: {final_save_path}")
    
    if is_main_process:
        swanlab.finish()

def main():
    # 0. 初始化统一的实验目录
    if not os.path.exists(EXPERIMENT_DIR):
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        # 子目录会在 TrainingArguments 中自动创建，或者手动创建
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    print(f"初始化实验目录: {EXPERIMENT_DIR}")
    
    # 1. 加载数据
    df = load_and_preprocess_data(DATASET_NAME, DATA_FILE, CACHE_DIR)
    
    # 2. 加载模型
    model, tokenizer = create_model_and_tokenizer(MODEL_NAME)
    
    # 3. 准备数据
    train_dataset, eval_dataset = prepare_dataset(df, tokenizer)
    
    # 4. 训练
    train_model(model, tokenizer, train_dataset, eval_dataset)

if __name__ == "__main__":
    main()