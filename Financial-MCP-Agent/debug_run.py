#!/usr/bin/env python3
"""
调试运行脚本 - Debug Runner

使用方法:
    python debug_run.py --query "分析茅台(600519)" --level verbose
    python debug_run.py --query "分析比亚迪" --level basic
    python debug_run.py --agent fundamental --stock sh.600519

参数说明:
    --query: 分析查询（完整分析）
    --agent: 指定单个智能体调试 (fundamental/technical/value/news)
    --stock: 股票代码（用于单个智能体测试）
    --level: 调试级别 (none/basic/detailed/verbose)
"""

import sys
import os
import asyncio
import argparse

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 导入调试配置
from debug_config import (
    DEBUG_CONFIG, DebugLevel, 
    init_debug_session, finalize_debug_session,
    debug_agent, log_debug, Colors
)

# 导入主程序
from src.main import main
from src.utils.state_definition import AgentState

# 导入各个智能体
from src.agents.fundamental_agent import fundamental_agent
from src.agents.technical_agent import technical_agent
from src.agents.value_agent import value_agent
from src.agents.news_agent import news_agent
from src.agents.summary_agent import summary_agent

async def debug_single_agent(agent_name: str, stock_code: str):
    """调试单个智能体"""
    
    # 准备初始状态
    initial_state = AgentState(
        messages=[],
        data={
            "stock_code": stock_code,
            "query": f"分析 {stock_code}",
            "company_name": stock_code.split('.')[1] if '.' in stock_code else stock_code,
            "current_date": "2025-11-23"
        },
        metadata={}
    )
    
    init_debug_session(f"调试智能体: {agent_name}", stock_code)
    
    # 选择智能体
    agent_map = {
        'fundamental': fundamental_agent,
        'technical': technical_agent,
        'value': value_agent,
        'news': news_agent,
        'summary': summary_agent
    }
    
    if agent_name not in agent_map:
        print(f"❌ 未知的智能体: {agent_name}")
        print(f"可用的智能体: {', '.join(agent_map.keys())}")
        return
    
    agent_func = agent_map[agent_name]
    
    # 包装智能体函数
    wrapped_agent = debug_agent(agent_name)(agent_func)
    
    try:
        # 执行智能体
        result = await wrapped_agent(initial_state)
        
        log_debug("\n" + "="*80, DebugLevel.BASIC, Colors.GREEN)
        log_debug("✅ 智能体执行成功!", DebugLevel.BASIC, Colors.GREEN)
        log_debug("="*80, DebugLevel.BASIC, Colors.GREEN)
        
        # 显示结果摘要
        if 'data' in result:
            for key, value in result['data'].items():
                if isinstance(value, str) and len(value) > 100:
                    log_debug(f"\n{key}: {value[:100]}...", DebugLevel.BASIC, Colors.CYAN)
                else:
                    log_debug(f"\n{key}: {value}", DebugLevel.BASIC, Colors.CYAN)
        
    except Exception as e:
        log_debug("\n" + "="*80, DebugLevel.BASIC, Colors.RED)
        log_debug(f"❌ 智能体执行失败: {e}", DebugLevel.BASIC, Colors.RED)
        log_debug("="*80, DebugLevel.BASIC, Colors.RED)
        raise
    finally:
        finalize_debug_session()

async def debug_full_analysis(query: str):
    """调试完整分析流程"""
    init_debug_session(query)
    
    try:
        # 运行主程序（已经集成了调试功能）
        await main()
        
        log_debug("\n" + "="*80, DebugLevel.BASIC, Colors.GREEN)
        log_debug("✅ 完整分析成功!", DebugLevel.BASIC, Colors.GREEN)
        log_debug("="*80, DebugLevel.BASIC, Colors.GREEN)
        
    except Exception as e:
        log_debug("\n" + "="*80, DebugLevel.BASIC, Colors.RED)
        log_debug(f"❌ 分析失败: {e}", DebugLevel.BASIC, Colors.RED)
        log_debug("="*80, DebugLevel.BASIC, Colors.RED)
        raise
    finally:
        finalize_debug_session()

def main_cli():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='金融分析智能体调试工具')
    
    # 主要参数
    parser.add_argument('--query', type=str, help='分析查询')
    parser.add_argument('--agent', type=str, choices=['fundamental', 'technical', 'value', 'news', 'summary'],
                       help='调试单个智能体')
    parser.add_argument('--stock', type=str, help='股票代码（用于单智能体测试）')
    
    # 调试级别
    parser.add_argument('--level', type=str, 
                       choices=['none', 'basic', 'detailed', 'verbose'],
                       default='detailed',
                       help='调试级别')
    
    # 其他选项
    parser.add_argument('--no-color', action='store_true', help='禁用彩色输出')
    parser.add_argument('--no-save', action='store_true', help='不保存中间状态')
    parser.add_argument('--no-perf', action='store_true', help='不跟踪性能')
    
    args = parser.parse_args()
    
    # 设置调试级别
    level_map = {
        'none': DebugLevel.NONE,
        'basic': DebugLevel.BASIC,
        'detailed': DebugLevel.DETAILED,
        'verbose': DebugLevel.VERBOSE
    }
    DEBUG_CONFIG['level'] = level_map[args.level]
    DEBUG_CONFIG['colored_output'] = not args.no_color
    DEBUG_CONFIG['save_intermediate_states'] = not args.no_save
    DEBUG_CONFIG['track_performance'] = not args.no_perf
    
    # 执行调试
    if args.agent:
        # 单智能体调试
        if not args.stock:
            print("❌ 调试单个智能体需要提供 --stock 参数")
            sys.exit(1)
        asyncio.run(debug_single_agent(args.agent, args.stock))
    elif args.query:
        # 完整分析调试
        # 临时设置命令行参数
        sys.argv = ['debug_run.py', '--command', args.query]
        asyncio.run(debug_full_analysis(args.query))
    else:
        parser.print_help()
        print("\n示例:")
        print("  python debug_run.py --query '分析茅台(600519)' --level verbose")
        print("  python debug_run.py --agent fundamental --stock sh.600519 --level detailed")

if __name__ == '__main__':
    main_cli()
