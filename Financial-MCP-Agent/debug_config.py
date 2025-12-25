"""
调试配置模块 - Debug Configuration Module

提供详细的调试功能，包括：
1. 步骤跟踪 - 跟踪每一步执行
2. 性能监控 - 监控每个智能体的执行时间
3. 数据流可视化 - 显示数据在智能体间的流动
4. 错误诊断 - 详细的错误堆栈和上下文
"""

import time
import json
import traceback
from functools import wraps
from datetime import datetime
from typing import Any, Dict, Callable
import os

# 调试级别配置
class DebugLevel:
    NONE = 0      # 无调试输出
    BASIC = 1     # 基本信息：智能体开始/结束
    DETAILED = 2  # 详细信息：包含输入/输出
    VERBOSE = 3   # 冗长模式：包含所有细节

# 全局调试配置
DEBUG_CONFIG = {
    'enabled': False,
    'level': DebugLevel.VERBOSE,
    'log_file': None,
    'track_performance': True,
    'show_data_flow': True,
    'colored_output': True,
    'save_intermediate_states': True,
}

# ANSI颜色代码
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colored(text: str, color: str) -> str:
    """为文本添加颜色"""
    if DEBUG_CONFIG['colored_output']:
        return f"{color}{text}{Colors.ENDC}"
    return text

# 性能统计
performance_stats = {}

def log_debug(message: str, level: int = DebugLevel.BASIC, color: str = Colors.CYAN):
    """调试日志输出"""
    if not DEBUG_CONFIG['enabled'] or DEBUG_CONFIG['level'] < level:
        return
    
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    formatted_msg = f"[{timestamp}] {message}"
    print(colored(formatted_msg, color))
    
    if DEBUG_CONFIG['log_file']:
        with open(DEBUG_CONFIG['log_file'], 'a', encoding='utf-8') as f:
            f.write(f"{formatted_msg}\n")

def print_separator(char='=', length=80, color=Colors.BLUE):
    """打印分隔线"""
    log_debug(char * length, DebugLevel.BASIC, color)

def print_section_header(title: str):
    """打印章节标题"""
    print_separator('=', 80, Colors.HEADER)
    log_debug(f"  {title}", DebugLevel.BASIC, Colors.HEADER + Colors.BOLD)
    print_separator('=', 80, Colors.HEADER)

def debug_agent(agent_name: str):
    """
    装饰器：为智能体函数添加调试功能
    
    Args:
        agent_name: 智能体名称
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            # 记录开始时间
            start_time = time.time()
            
            # 打印智能体启动信息
            print_section_header(f"🤖 启动智能体: {agent_name}")
            
            # 显示输入状态（详细模式）
            if DEBUG_CONFIG['level'] >= DebugLevel.DETAILED:
                log_debug(f"\n📥 输入状态 ({agent_name}):", DebugLevel.DETAILED, Colors.CYAN)
                log_debug(f"  查询: {state.get('data', {}).get('query', 'N/A')}", DebugLevel.DETAILED, Colors.CYAN)
                log_debug(f"  股票代码: {state.get('data', {}).get('stock_code', 'N/A')}", DebugLevel.DETAILED, Colors.CYAN)
                log_debug(f"  公司名称: {state.get('data', {}).get('company_name', 'N/A')}", DebugLevel.DETAILED, Colors.CYAN)
                
                # 显示已有的分析结果
                data_keys = [k for k in state.get('data', {}).keys() if 'analysis' in k]
                if data_keys:
                    log_debug(f"  已完成的分析: {', '.join(data_keys)}", DebugLevel.DETAILED, Colors.GREEN)
            
            try:
                # 执行智能体函数
                log_debug(f"\n⚙️  执行 {agent_name}...", DebugLevel.BASIC, Colors.YELLOW)
                result = await func(state)
                
                # 记录结束时间
                end_time = time.time()
                duration = end_time - start_time
                
                # 性能统计
                if DEBUG_CONFIG['track_performance']:
                    performance_stats[agent_name] = {
                        'duration': duration,
                        'timestamp': datetime.now().isoformat()
                    }
                
                # 显示输出状态（详细模式）
                if DEBUG_CONFIG['level'] >= DebugLevel.DETAILED:
                    log_debug(f"\n📤 输出状态 ({agent_name}):", DebugLevel.DETAILED, Colors.GREEN)
                    
                    # 检查新增的分析结果
                    new_keys = [k for k in result.get('data', {}).keys() 
                               if k not in state.get('data', {}).keys()]
                    if new_keys:
                        log_debug(f"  新增数据字段: {', '.join(new_keys)}", DebugLevel.DETAILED, Colors.GREEN)
                        
                        # 显示部分内容（冗长模式）
                        if DEBUG_CONFIG['level'] >= DebugLevel.VERBOSE:
                            for key in new_keys:
                                content = str(result['data'].get(key, ''))
                                preview = content[:200] + '...' if len(content) > 200 else content
                                log_debug(f"\n  {key} 内容预览:", DebugLevel.VERBOSE, Colors.CYAN)
                                log_debug(f"    {preview}", DebugLevel.VERBOSE, Colors.CYAN)
                
                # 显示执行时间
                log_debug(f"\n✅ {agent_name} 完成 - 耗时: {duration:.2f}秒", DebugLevel.BASIC, Colors.GREEN)
                print_separator('-', 80, Colors.GREEN)
                
                # 保存中间状态
                if DEBUG_CONFIG['save_intermediate_states']:
                    save_intermediate_state(agent_name, result)
                
                return result
                
            except Exception as e:
                # 记录错误
                end_time = time.time()
                duration = end_time - start_time
                
                log_debug(f"\n❌ {agent_name} 失败 - 耗时: {duration:.2f}秒", DebugLevel.BASIC, Colors.RED)
                log_debug(f"错误类型: {type(e).__name__}", DebugLevel.BASIC, Colors.RED)
                log_debug(f"错误信息: {str(e)}", DebugLevel.BASIC, Colors.RED)
                
                if DEBUG_CONFIG['level'] >= DebugLevel.DETAILED:
                    log_debug(f"\n堆栈跟踪:", DebugLevel.DETAILED, Colors.RED)
                    log_debug(traceback.format_exc(), DebugLevel.DETAILED, Colors.RED)
                
                print_separator('-', 80, Colors.RED)
                raise
        
        @wraps(func)
        def sync_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            # 同步版本（如果需要）
            import asyncio
            return asyncio.run(async_wrapper(state))
        
        # 根据原函数是否为协程返回对应的包装器
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def save_intermediate_state(agent_name: str, state: Dict[str, Any]):
    """保存中间状态到文件"""
    try:
        log_dir = "/home/data1/gjl/more_learning/shock_invest_Agent/Finance/Financial-MCP-Agent/debug_states"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{log_dir}/{timestamp}_{agent_name}_state.json"
        
        # 简化状态以便保存
        simplified_state = {
            'data': {k: str(v)[:500] if isinstance(v, str) and len(str(v)) > 500 else v 
                    for k, v in state.get('data', {}).items()},
            'metadata': state.get('metadata', {}),
            'messages': [str(m)[:200] for m in state.get('messages', [])]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(simplified_state, f, ensure_ascii=False, indent=2)
        
        log_debug(f"💾 中间状态已保存: {filename}", DebugLevel.VERBOSE, Colors.BLUE)
    except Exception as e:
        log_debug(f"⚠️  保存中间状态失败: {e}", DebugLevel.BASIC, Colors.YELLOW)

def print_performance_summary():
    """打印性能摘要"""
    if not performance_stats or not DEBUG_CONFIG['track_performance']:
        return
    
    print_section_header("📊 性能统计摘要")
    
    total_time = sum(stat['duration'] for stat in performance_stats.values())
    
    log_debug(f"\n总执行时间: {total_time:.2f}秒\n", DebugLevel.BASIC, Colors.BOLD)
    
    # 按时间排序
    sorted_stats = sorted(performance_stats.items(), key=lambda x: x[1]['duration'], reverse=True)
    
    for agent_name, stat in sorted_stats:
        duration = stat['duration']
        percentage = (duration / total_time * 100) if total_time > 0 else 0
        bar_length = int(percentage / 2)  # 50个字符表示100%
        bar = '█' * bar_length + '░' * (50 - bar_length)
        
        log_debug(f"{agent_name:20s} {bar} {duration:6.2f}s ({percentage:5.1f}%)", 
                 DebugLevel.BASIC, Colors.CYAN)
    
    print_separator('=', 80, Colors.BLUE)

def init_debug_session(query: str, stock_code: str = None):
    """初始化调试会话"""
    print_section_header("🚀 金融分析智能体系统 - 调试模式")
    
    log_debug(f"\n查询: {query}", DebugLevel.BASIC, Colors.GREEN)
    if stock_code:
        log_debug(f"股票代码: {stock_code}", DebugLevel.BASIC, Colors.GREEN)
    log_debug(f"调试级别: {DEBUG_CONFIG['level']}", DebugLevel.BASIC, Colors.YELLOW)
    log_debug(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", DebugLevel.BASIC, Colors.BLUE)
    
    # 设置日志文件
    if DEBUG_CONFIG['log_file'] is None:
        log_dir = "/home/data1/gjl/more_learning/shock_invest_Agent/Finance/Financial-MCP-Agent/debug_logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        DEBUG_CONFIG['log_file'] = f"{log_dir}/debug_{timestamp}.log"
        log_debug(f"日志文件: {DEBUG_CONFIG['log_file']}", DebugLevel.BASIC, Colors.BLUE)
    
    print_separator('=', 80, Colors.HEADER)

def finalize_debug_session():
    """结束调试会话"""
    print_performance_summary()
    log_debug(f"\n调试会话结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
             DebugLevel.BASIC, Colors.GREEN)
