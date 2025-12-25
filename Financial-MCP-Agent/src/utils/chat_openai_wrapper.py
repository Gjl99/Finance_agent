"""
ChatOpenAI包装器:修复工具调用参数格式问题

某些OpenAI兼容的API返回的tool_calls格式与标准OpenAI不同,导致Pydantic验证错误。
此包装器通过猴子补丁方式修复这个问题。
"""

import json
import logging
from typing import Any, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import _convert_dict_to_message as _original_convert
from langchain_core.language_models.chat_models import BaseChatModel

# 设置logger
logger = logging.getLogger(__name__)


def _patched_convert_dict_to_message(response: Dict[str, Any]):
    """
    修补后的转换函数,处理tool_calls格式问题
    
    关键修复:
    1. 确保 function.arguments 存在且为JSON字符串
    2. parse_tool_call 会从 function.arguments 解析出 args (字典)
    3. 最终 AIMessage.tool_calls 中的 args 必须是字典类型
    
    Args:
        response: 原始API响应字典
        
    Returns:
        转换后的AIMessage
    """
    # 深拷贝response以避免修改原始数据
    import copy
    response = copy.deepcopy(response)
    
    logger.debug(f"Processing response with role: {response.get('role')}")
    
    # 只处理assistant角色的消息且包含tool_calls
    if response.get("role") == "assistant" and "tool_calls" in response:
        tool_calls = response.get("tool_calls", [])
        logger.debug(f"Found {len(tool_calls)} tool calls")
        
        for i, raw_tool_call in enumerate(tool_calls):
            if not isinstance(raw_tool_call, dict):
                continue
            
            logger.debug(f"Tool call {i} before fix: {json.dumps(raw_tool_call, indent=2, ensure_ascii=False)[:200]}")
            
            # 确保有function字段
            if "function" not in raw_tool_call:
                raw_tool_call["function"] = {}
            
            # 处理简化的 args 格式
            # 某些API返回 args 而不是 function.arguments
            if "args" in raw_tool_call:
                args = raw_tool_call["args"]
                logger.info(f"Tool call {i} - Found 'args' field, type: {type(args).__name__}, converting to function.arguments")
                
                # 将 args 转换为 function.arguments (必须是JSON字符串)
                if isinstance(args, str):
                    # 验证是否为有效JSON
                    try:
                        json.loads(args)  # 验证
                        raw_tool_call["function"]["arguments"] = args
                        logger.info(f"Tool call {i} - ✓ Moved args (string) to function.arguments")
                    except json.JSONDecodeError as e:
                        logger.error(f"Tool call {i} - Invalid JSON in args: {e}")
                        raw_tool_call["function"]["arguments"] = "{}"
                elif isinstance(args, dict):
                    # 字典格式,转为JSON字符串
                    raw_tool_call["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
                    logger.info(f"Tool call {i} - ✓ Converted args (dict) to function.arguments (JSON string)")
                else:
                    logger.warning(f"Tool call {i} - Unexpected args type: {type(args)}")
                    raw_tool_call["function"]["arguments"] = "{}"
                
                # 删除 args 字段,避免干扰
                del raw_tool_call["args"]
            
            # 确保 function.arguments 存在且为字符串
            if "function" in raw_tool_call:
                function = raw_tool_call["function"]
                
                if "arguments" not in function:
                    function["arguments"] = "{}"
                    logger.debug(f"Tool call {i} - Set default empty arguments")
                elif not isinstance(function["arguments"], str):
                    # arguments不是字符串,转换之
                    if isinstance(function["arguments"], dict):
                        function["arguments"] = json.dumps(function["arguments"], ensure_ascii=False)
                        logger.info(f"Tool call {i} - ✓ Converted function.arguments from dict to string")
                    else:
                        logger.warning(f"Tool call {i} - Unexpected arguments type: {type(function['arguments'])}")
                        function["arguments"] = "{}"
                
                # 检查是否双重编码 (Double Encoding)
                try:
                    parsed_args = json.loads(function["arguments"])
                    if isinstance(parsed_args, str):
                        try:
                            inner_args = json.loads(parsed_args)
                            if isinstance(inner_args, dict):
                                function["arguments"] = parsed_args
                                logger.info(f"Tool call {i} - ✓ Fixed double encoded function.arguments")
                        except json.JSONDecodeError:
                            pass
                except json.JSONDecodeError:
                    pass

                # 确保有name字段
                if "name" not in function:
                    function["name"] = raw_tool_call.get("name", "unknown_function")
            
            logger.debug(f"Tool call {i} after fix: function.arguments = {raw_tool_call['function'].get('arguments', 'N/A')[:100]}")
    
    # 调用原始的转换函数
    try:
        logger.debug("Calling original _convert_dict_to_message")
        result = _original_convert(response)
        logger.debug(f"✓ Successfully converted to {type(result).__name__}")
        
        # 验证结果
        if hasattr(result, 'tool_calls') and result.tool_calls:
            for i, tc in enumerate(result.tool_calls):
                if isinstance(tc, dict) and 'args' in tc:
                    args_type = type(tc['args']).__name__
                    logger.debug(f"Result tool_call {i} - args type: {args_type}")
                    if not isinstance(tc['args'], dict):
                        logger.error(f"❌ Result tool_call {i} - args is NOT dict! type={args_type}, value={str(tc['args'])[:100]}")
                    else:
                        logger.info(f"✓ Result tool_call {i} - args is dict (correct!)")
        
        return result
    except Exception as e:
        logger.error(f"❌ Error in original_convert: {e}")
        logger.error(f"Response structure:\n{json.dumps(response, indent=2, ensure_ascii=False)}")
        raise


def create_fixed_chat_openai(
    model: str,
    temperature: float = 0.7,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> BaseChatModel:
    """
    创建修复了工具调用问题的ChatOpenAI实例
    
    修复逻辑:
    1. API返回 args (字符串/字典) -> 转为 function.arguments (JSON字符串)
    2. parse_tool_call 从 function.arguments -> 解析为 args (字典)
    3. AIMessage 验证 args 为字典 -> ✓ 成功
    
    Args:
        model: 模型名称
        temperature: 温度参数
        base_url: API基础URL
        api_key: API密钥
        max_tokens: 最大token数
        **kwargs: 其他参数
        
    Returns:
        修复后的ChatOpenAI实例
    """
    import langchain_openai.chat_models.base as openai_base
    
    # 应用猴子补丁
    openai_base._convert_dict_to_message = _patched_convert_dict_to_message
    logger.info("✓ Applied monkey patch to langchain_openai.chat_models.base._convert_dict_to_message")
    
    # 创建ChatOpenAI实例
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        **kwargs
    )
    
    logger.info(f"✓ Created fixed ChatOpenAI instance for model: {model}")
    return llm
