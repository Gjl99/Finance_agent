"""
调试工具模块：提供详细的调试和追踪功能

使用方法：
1. 在代码中导入：from src.utils.debug_tools import debug_trace, enable_debug_mode
2. 启用调试模式：enable_debug_mode()
3. 在关键位置添加追踪点：debug_trace("description", local_vars)
"""

import os
import sys
import json
import functools
import traceback
from typing import Any, Dict, Callable
from datetime import datetime

# 调试模式开关
_DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
_DEBUG_LOG_FILE = None


def enable_debug_mode(log_file: str = None):
    """启用详细调试模式"""
    global _DEBUG_MODE, _DEBUG_LOG_FILE
    _DEBUG_MODE = True
    if log_file:
        _DEBUG_LOG_FILE = log_file
        print(f"✓ 调试模式已启用，日志文件: {log_file}")
    else:
        print("✓ 调试模式已启用，输出到控制台")


def disable_debug_mode():
    """禁用调试模式"""
    global _DEBUG_MODE
    _DEBUG_MODE = False
    print("✓ 调试模式已禁用")


def debug_trace(message: str, data: Dict[str, Any] = None, level: str = "INFO"):
    """
    调试追踪点
    
    Args:
        message: 调试消息
        data: 要记录的数据
        level: 日志级别 (INFO, DEBUG, WARNING, ERROR)
    """
    if not _DEBUG_MODE:
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    caller_frame = traceback.extract_stack()[-2]
    location = f"{caller_frame.filename}:{caller_frame.lineno} ({caller_frame.name})"
    
    output = f"\n{'='*80}\n"
    output += f"[{timestamp}] [{level}] {message}\n"
    output += f"位置: {location}\n"
    
    if data:
        output += "数据:\n"
        for key, value in data.items():
            try:
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, ensure_ascii=False, indent=2)
                else:
                    value_str = str(value)
                # 限制输出长度
                if len(value_str) > 500:
                    value_str = value_str[:500] + "... (截断)"
                output += f"  {key}: {value_str}\n"
            except Exception as e:
                output += f"  {key}: <无法序列化: {e}>\n"
    
    output += f"{'='*80}\n"
    
    if _DEBUG_LOG_FILE:
        with open(_DEBUG_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(output)
    else:
        print(output)


def trace_function(func: Callable) -> Callable:
    """
    函数追踪装饰器
    
    使用方法：
        @trace_function
        def my_function(arg1, arg2):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _DEBUG_MODE:
            return func(*args, **kwargs)
        
        func_name = f"{func.__module__}.{func.__name__}"
        debug_trace(
            f"进入函数: {func_name}",
            {
                "args": args,
                "kwargs": kwargs
            },
            "DEBUG"
        )
        
        try:
            result = func(*args, **kwargs)
            debug_trace(
                f"函数返回: {func_name}",
                {"result_type": type(result).__name__},
                "DEBUG"
            )
            return result
        except Exception as e:
            debug_trace(
                f"函数异常: {func_name}",
                {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                },
                "ERROR"
            )
            raise
    
    return wrapper


async def trace_async_function(func: Callable) -> Callable:
    """
    异步函数追踪装饰器
    
    使用方法：
        @trace_async_function
        async def my_async_function(arg1, arg2):
            ...
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        if not _DEBUG_MODE:
            return await func(*args, **kwargs)
        
        func_name = f"{func.__module__}.{func.__name__}"
        debug_trace(
            f"进入异步函数: {func_name}",
            {
                "args": args,
                "kwargs": kwargs
            },
            "DEBUG"
        )
        
        try:
            result = await func(*args, **kwargs)
            debug_trace(
                f"异步函数返回: {func_name}",
                {"result_type": type(result).__name__},
                "DEBUG"
            )
            return result
        except Exception as e:
            debug_trace(
                f"异步函数异常: {func_name}",
                {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                },
                "ERROR"
            )
            raise
    
    return wrapper


def print_state(state: Dict[str, Any], title: str = "当前状态"):
    """
    打印 AgentState 的详细信息
    
    Args:
        state: AgentState 字典
        title: 标题
    """
    if not _DEBUG_MODE:
        return
    
    debug_trace(
        title,
        {
            "data_keys": list(state.get("data", {}).keys()) if "data" in state else [],
            "messages_count": len(state.get("messages", [])),
            "metadata_keys": list(state.get("metadata", {}).keys()) if "metadata" in state else [],
            "sample_data": {k: str(v)[:100] for k, v in list(state.get("data", {}).items())[:5]}
        }
    )


def is_debug_mode() -> bool:
    """检查是否处于调试模式"""
    return _DEBUG_MODE
