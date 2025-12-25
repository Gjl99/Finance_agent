"""
金融分析智能体系统 - Web前端界面 (Financial Analysis AI Agent System - Web Frontend)

本文件实现了金融分析智能体系统的Web前端界面，提供以下功能：

1. Web界面：基于Flask的用户友好Web界面
2. 实时分析：支持股票代码和公司名称的实时分析
3. 报告展示：完整显示技术分析报告和其他分析结果
4. 响应式设计：适配不同屏幕尺寸的响应式Web设计
5. 实时状态：显示分析进度和实时状态更新
"""

import os
import sys
import asyncio
import json
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# 导入主要的分析系统
from main import main as run_analysis
from src.utils.logging_config import setup_logger
from src.utils.state_definition import AgentState
from src.utils.execution_logger import initialize_execution_logger, finalize_execution_logger

# 智能体模块导入
from src.agents.summary_agent import summary_agent
from src.agents.value_agent import value_agent
from src.agents.technical_agent import technical_agent
from src.agents.fundamental_agent import fundamental_agent
from src.agents.news_agent import news_agent

# LangGraph工作流框架导入
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import re
import baostock as bs
import pandas as pd

# 加载环境变量
load_dotenv(override=True)

# 设置日志
logger = setup_logger(__name__)

# 自定义日志处理器，用于将日志发送到WebSocket
class SocketLogHandler(logging.Handler):
    def __init__(self, socketio):
        super().__init__()
        self.socketio = socketio

    def emit(self, record):
        try:
            msg = self.format(record)
            self.socketio.emit('log_message', {'message': msg, 'type': 'log'})
        except Exception:
            # 避免日志处理器内部错误导致程序崩溃
            pass

# 自定义标准输出重定向，用于捕获print输出
class SocketStdout:
    def __init__(self, socketio, original_stdout):
        self.socketio = socketio
        self.original_stdout = original_stdout

    def write(self, text):
        try:
            self.original_stdout.write(text) # 同时也输出到控制台
            if text.strip():
                self.socketio.emit('log_message', {'message': text.strip(), 'type': 'stdout'})
        except Exception as e:
            # 确保即使WebSocket发送失败，控制台输出也能正常工作
            self.original_stdout.write(f"[SocketStdout Error] {e}\n")

    def flush(self):
        try:
            self.original_stdout.flush()
        except Exception:
            pass

# 创建Flask应用
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
app.config['SECRET_KEY'] = 'financial_analysis_secret_key'

# 创建SocketIO实例用于实时通信
# 强制使用 threading 模式以避免与 asyncio 冲突，并提高兼容性
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局变量存储分析状态
analysis_status = {
    'running': False,
    'current_stage': '',
    'progress': 0,
    'results': {},
    'error': None
}

def extract_stock_info(query):
    """从查询中提取股票代码和公司名称"""
    stock_code = None
    company_name = None
    
    # 使用与main.py相同的提取逻辑
    patterns = [
        # 模式1: 包含"请帮我分析一下"的复杂查询
        (r'请帮我分析一下\s*([^（(]+?)\s*[（(](\d{5,6})[)）]', 'name_code'),
        # 模式2: 包含"分析一下"的复杂查询
        (r'分析一下\s*([^（(]+?)\s*[（(](\d{5,6})[)）]', 'name_code'),
        # 模式3: 股票代码在括号内
        (r'分析\s*([^（(]+?)\s*[（(](\d{5,6})[)）]', 'name_code'),
        # 模式4: 直接包含5-6位数字股票代码
        (r'\b(\d{5,6})\b', 'code_only'),
        # 模式5: 分析公司名称
        (r'分析\s*([^0-9（）()\s]+)', 'name_only'),
        # 模式6: 包含"股票"关键词的查询
        (r'([^0-9（）()\s]+)\s*(?:这只|这个|的)?\s*股票', 'name_only'),
    ]
    
    for pattern, match_type in patterns:
        match = re.search(pattern, query)
        if match:
            if match_type == 'name_code':
                company_name = match.group(1).strip()
                stock_code = match.group(2)
                break
            elif match_type == 'code_only' and not stock_code:
                stock_code = match.group(1)
            elif match_type == 'name_only' and not company_name:
                company_name = match.group(1).strip()
    
    # 清理公司名称
    if company_name:
        stop_words = ['的', '这个', '这只', '一下', '看看', '了解', '分析', '帮我', '我想', '给我']
        for word in stop_words:
            company_name = company_name.replace(word, '').strip()
        if len(company_name) < 2:
            company_name = None
    
    return company_name, stock_code

def search_stock_code(company_name):
    """根据公司名称查找股票代码"""
    try:
        print(f"正在尝试查找 {company_name} 的股票代码...")
        # 登录系统
        lg = bs.login()
        if lg.error_code != '0':
            print(f"Baostock登录失败: {lg.error_msg}")
            return None
            
        # 获取所有A股股票列表
        rs = bs.query_stock_basic()
        if rs.error_code != '0':
            print(f"查询股票列表失败: {rs.error_msg}")
            bs.logout()
            return None
            
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
            
        bs.logout()
        
        if not data_list:
            return None
            
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 1. 精确匹配名称
        exact_match = df[df['code_name'] == company_name]
        if not exact_match.empty:
            code = exact_match.iloc[0]['code']
            print(f"找到精确匹配: {company_name} -> {code}")
            return code
            
        # 2. 包含匹配
        contains_match = df[df['code_name'].str.contains(company_name)]
        if not contains_match.empty:
            # 优先返回主板股票，这里简单返回第一个
            code = contains_match.iloc[0]['code']
            name = contains_match.iloc[0]['code_name']
            print(f"找到模糊匹配: {company_name} -> {name} ({code})")
            return code
            
        print(f"未找到 {company_name} 对应的股票代码")
        return None
    except Exception as e:
        print(f"查找股票代码出错: {e}")
        return None

async def run_financial_analysis(user_query):
    """运行金融分析工作流"""
    global analysis_status
    
    # 设置日志捕获
    log_handler = SocketLogHandler(socketio)
    log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    
    # 设置stdout捕获
    original_stdout = sys.stdout
    sys.stdout = SocketStdout(socketio, original_stdout)
    
    try:
        analysis_status['running'] = True
        analysis_status['progress'] = 10
        analysis_status['current_stage'] = '初始化分析系统...'
        socketio.emit('status_update', analysis_status)
        
        # 初始化执行日志系统
        execution_logger = initialize_execution_logger()
        
        # 构建工作流
        workflow = StateGraph(AgentState)
        workflow.add_node("start_node", lambda state: state)
        workflow.add_node("fundamental_analyst", fundamental_agent)
        workflow.add_node("technical_analyst", technical_agent)
        workflow.add_node("value_analyst", value_agent)
        workflow.add_node("news_analyst", news_agent)
        workflow.add_node("summarizer", summary_agent)
        
        workflow.set_entry_point("start_node")
        workflow.add_edge("start_node", "fundamental_analyst")
        workflow.add_edge("start_node", "technical_analyst")
        workflow.add_edge("start_node", "value_analyst")
        workflow.add_edge("start_node", "news_analyst")
        workflow.add_edge("fundamental_analyst", "summarizer")
        workflow.add_edge("technical_analyst", "summarizer")
        workflow.add_edge("value_analyst", "summarizer")
        workflow.add_edge("news_analyst", "summarizer")
        workflow.add_edge("summarizer", END)
        
        app_workflow = workflow.compile()
        
        analysis_status['progress'] = 20
        analysis_status['current_stage'] = '提取股票信息...'
        socketio.emit('status_update', analysis_status)
        
        # 提取股票信息
        company_name, stock_code = extract_stock_info(user_query)
        
        # 如果没有提取到代码但有公司名，尝试自动查找
        if not stock_code and company_name:
            found_code = search_stock_code(company_name)
            if found_code:
                # baostock返回的格式通常是 sh.600000 或 sz.000001
                # 我们需要根据后续逻辑决定是否保留前缀
                # 下面的逻辑会再次添加前缀，所以如果已经有前缀，需要处理
                if found_code.startswith('sh.') or found_code.startswith('sz.'):
                    stock_code = found_code.split('.')[1]
                else:
                    stock_code = found_code
                
                socketio.emit('log_message', {'message': f'已自动识别股票代码: {stock_code} ({company_name})', 'type': 'system'})
        
        # 准备时间信息
        current_datetime = datetime.now()
        current_date_cn = current_datetime.strftime("%Y年%m月%d日")
        current_date_en = current_datetime.strftime("%Y-%m-%d")
        current_weekday_cn = ["星期一", "星期二", "星期三", "星期四",
                              "星期五", "星期六", "星期日"][current_datetime.weekday()]
        current_time = current_datetime.strftime("%H:%M:%S")
        current_time_info = f"{current_date_cn} ({current_date_en}) {current_weekday_cn} {current_time}"
        
        # 准备初始状态
        initial_data = {
            "query": user_query,
            "current_date": current_date_en,
            "current_date_cn": current_date_cn,
            "current_time": current_time,
            "current_weekday_cn": current_weekday_cn,
            "current_time_info": current_time_info,
            "analysis_timestamp": current_datetime.isoformat()
        }
        
        if company_name:
            initial_data["company_name"] = company_name
        if stock_code:
            if stock_code.startswith('6'):
                initial_data["stock_code"] = f"sh.{stock_code}"
            elif stock_code.startswith('0') or stock_code.startswith('3'):
                initial_data["stock_code"] = f"sz.{stock_code}"
            else:
                initial_data["stock_code"] = stock_code
        
        initial_state = AgentState(
            messages=[],
            data=initial_data,
            metadata={}
        )
        
        analysis_status['progress'] = 30
        analysis_status['current_stage'] = '正在执行基本面分析...'
        socketio.emit('status_update', analysis_status)
        
        # 执行工作流
        final_state = await app_workflow.ainvoke(initial_state)
        
        analysis_status['progress'] = 90
        analysis_status['current_stage'] = '整理分析结果...'
        socketio.emit('status_update', analysis_status)
        
        # 处理结果
        if final_state and final_state.get("data"):
            analysis_status['results'] = {
                'query': user_query,
                'company_name': company_name,
                'stock_code': stock_code,
                'analysis_time': current_time_info,
                'final_report': final_state["data"].get("final_report", ""),
                'fundamental_analysis': final_state["data"].get("fundamental_analysis", ""),
                'technical_analysis': final_state["data"].get("technical_analysis", ""),
                'value_analysis': final_state["data"].get("value_analysis", ""),
                'news_analysis': final_state["data"].get("news_analysis", ""),
                'report_path': final_state["data"].get("report_path", "")
            }
        
        analysis_status['progress'] = 100
        analysis_status['current_stage'] = '分析完成!'
        analysis_status['running'] = False
        socketio.emit('status_update', analysis_status)
        socketio.emit('analysis_complete', analysis_status['results'])
        
        finalize_execution_logger(success=True)
        
    except Exception as e:
        analysis_status['error'] = str(e)
        analysis_status['running'] = False
        analysis_status['current_stage'] = f'分析出错: {str(e)}'
        socketio.emit('status_update', analysis_status)
        finalize_execution_logger(success=False, error=str(e))
        logger.error(f"Analysis error: {e}", exc_info=True)
    finally:
        # 清理日志捕获和stdout重定向
        root_logger.removeHandler(log_handler)
        sys.stdout = original_stdout

@app.route('/')
def index():
    """主页路由"""
    return render_template('index.html')

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    """开始分析的API端点"""
    global analysis_status
    
    if analysis_status['running']:
        return jsonify({'error': '分析正在进行中，请等待完成'}), 400
    
    user_query = request.json.get('query', '').strip()
    if not user_query:
        return jsonify({'error': '请输入分析查询'}), 400
    
    # 重置状态
    analysis_status = {
        'running': True,
        'current_stage': '准备开始分析...',
        'progress': 0,
        'results': {},
        'error': None
    }
    
    # 在新线程中运行异步分析
    def run_analysis_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_financial_analysis(user_query))
    
    thread = threading.Thread(target=run_analysis_thread)
    thread.start()
    
    return jsonify({'message': '分析已开始', 'status': 'started'})

@app.route('/analysis_status')
def get_analysis_status():
    """获取分析状态的API端点"""
    return jsonify(analysis_status)

@socketio.on('connect')
def handle_connect():
    """WebSocket连接处理"""
    emit('connected', {'data': '已连接到服务器'})
    # 发送一条测试日志，确认连接正常
    emit('log_message', {'message': '系统连接成功，准备就绪...', 'type': 'system'})

@socketio.on('disconnect')
def handle_disconnect():
    """WebSocket断开连接处理"""
    logger.info('客户端断开连接')

if __name__ == '__main__':
    # 确保模板和静态文件目录存在
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    logger.info("启动金融分析Web界面...")
    logger.info("访问地址: http://localhost:5000")
    
    # 启动Flask-SocketIO应用
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)