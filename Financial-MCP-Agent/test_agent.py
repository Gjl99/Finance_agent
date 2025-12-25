"""
简化的测试脚本：用于调试单个 Agent 或工作流组件

使用方法：
    python test_agent.py --agent fundamental --stock sh.600519
    python test_agent.py --agent technical --stock sz.000001
    python test_agent.py --workflow  # 测试完整工作流
"""

import os
import sys
import asyncio
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv
from src.utils.state_definition import AgentState
from src.utils.logging_config import setup_logger
from src.utils.debug_tools import enable_debug_mode, debug_trace, print_state

# 加载环境变量
load_dotenv(override=True)

# 启用调试模式
enable_debug_mode("debug_logs/test_agent.log")

logger = setup_logger(__name__)


async def test_fundamental_agent(stock_code: str):
    """测试基本面分析 Agent"""
    from src.agents.fundamental_agent import fundamental_agent
    
    debug_trace("开始测试基本面分析 Agent", {"stock_code": stock_code})
    
    initial_state = AgentState(
        messages=[],
        data={
            "query": f"分析{stock_code}的基本面",
            "stock_code": stock_code,
            "company_name": "测试公司",
            "current_date": "2025-11-17",
            "current_time_info": "2025年11月17日"
        },
        metadata={}
    )
    
    print_state(initial_state, "初始状态")
    
    result = await fundamental_agent(initial_state)
    
    print_state(result, "结果状态")
    
    if "fundamental_analysis" in result.get("data", {}):
        print("\n✓ 基本面分析结果:")
        print(result["data"]["fundamental_analysis"][:500])
    else:
        print("\n✗ 未生成基本面分析")


async def test_technical_agent(stock_code: str):
    """测试技术分析 Agent"""
    from src.agents.technical_agent import technical_agent
    
    debug_trace("开始测试技术分析 Agent", {"stock_code": stock_code})
    
    initial_state = AgentState(
        messages=[],
        data={
            "query": f"分析{stock_code}的技术面",
            "stock_code": stock_code,
            "company_name": "测试公司",
            "current_date": "2025-11-17",
            "current_time_info": "2025年11月17日"
        },
        metadata={}
    )
    
    print_state(initial_state, "初始状态")
    
    result = await technical_agent(initial_state)
    
    print_state(result, "结果状态")
    
    if "technical_analysis" in result.get("data", {}):
        print("\n✓ 技术分析结果:")
        print(result["data"]["technical_analysis"][:500])
    else:
        print("\n✗ 未生成技术分析")


async def test_value_agent(stock_code: str):
    """测试估值分析 Agent"""
    from src.agents.value_agent import value_agent
    
    debug_trace("开始测试估值分析 Agent", {"stock_code": stock_code})
    
    initial_state = AgentState(
        messages=[],
        data={
            "query": f"分析{stock_code}的估值",
            "stock_code": stock_code,
            "company_name": "测试公司",
            "current_date": "2025-11-17",
            "current_time_info": "2025年11月17日"
        },
        metadata={}
    )
    
    print_state(initial_state, "初始状态")
    
    result = await value_agent(initial_state)
    
    print_state(result, "结果状态")
    
    if "value_analysis" in result.get("data", {}):
        print("\n✓ 估值分析结果:")
        print(result["data"]["value_analysis"][:500])
    else:
        print("\n✗ 未生成估值分析")


async def test_news_agent(stock_code: str):
    """测试新闻分析 Agent"""
    from src.agents.news_agent import news_agent
    
    debug_trace("开始测试新闻分析 Agent", {"stock_code": stock_code})
    
    initial_state = AgentState(
        messages=[],
        data={
            "query": f"分析{stock_code}的新闻",
            "stock_code": stock_code,
            "company_name": "测试公司",
            "current_date": "2025-11-17",
            "current_time_info": "2025年11月17日"
        },
        metadata={}
    )
    
    print_state(initial_state, "初始状态")
    
    result = await news_agent(initial_state)
    
    print_state(result, "结果状态")
    
    if "news_analysis" in result.get("data", {}):
        print("\n✓ 新闻分析结果:")
        print(result["data"]["news_analysis"][:500])
    else:
        print("\n✗ 未生成新闻分析")


async def test_workflow(stock_code: str):
    """测试完整工作流"""
    from src.main import main
    
    debug_trace("开始测试完整工作流", {"stock_code": stock_code})
    
    # 设置命令行参数
    sys.argv = ["test_agent.py", "--command", f"分析{stock_code}"]
    
    await main()


async def main():
    parser = argparse.ArgumentParser(description="Agent 测试工具")
    parser.add_argument("--agent", choices=["fundamental", "technical", "value", "news"], 
                       help="要测试的 Agent")
    parser.add_argument("--stock", default="sh.600519", help="股票代码 (默认: sh.600519)")
    parser.add_argument("--workflow", action="store_true", help="测试完整工作流")
    
    args = parser.parse_args()
    
    if args.workflow:
        await test_workflow(args.stock)
    elif args.agent == "fundamental":
        await test_fundamental_agent(args.stock)
    elif args.agent == "technical":
        await test_technical_agent(args.stock)
    elif args.agent == "value":
        await test_value_agent(args.stock)
    elif args.agent == "news":
        await test_news_agent(args.stock)
    else:
        print("请指定 --agent 或 --workflow")
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
