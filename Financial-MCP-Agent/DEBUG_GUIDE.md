# 调试指南

## 方法一：使用 VS Code 调试器（推荐）

### 步骤：

1. **打开 VS Code**
   ```bash
   code /home/data1/gjl/more_learning/shock_invest_Agent/Finance/Financial-MCP-Agent
   ```

2. **设置断点**
   - 在你想要调试的代码行左侧点击，设置红色断点
   - 推荐断点位置：
     * `src/main.py` 第 115 行（工作流创建）
     * `src/main.py` 第 445 行（工作流执行）
     * `src/agents/fundamental_agent.py` 第 95 行（LLM 创建）
     * `src/agents/fundamental_agent.py` 第 170 行（Agent 调用）

3. **启动调试**
   - 按 `F5` 或点击侧边栏的"运行和调试"
   - 选择 "Debug Financial Agent"
   - 程序会在断点处暂停

4. **调试操作**
   - `F5`: 继续执行
   - `F10`: 单步跳过
   - `F11`: 单步进入
   - `Shift+F11`: 单步跳出
   - 在"变量"面板查看所有变量值
   - 在"调用堆栈"面板查看函数调用链

### 高级技巧：

- **条件断点**：右键断点 → 编辑断点 → 添加条件（如 `stock_code == "sh.600519"`）
- **日志点**：右键 → 添加日志点，无需修改代码即可打印变量
- **监视表达式**：在"监视"面板添加表达式，实时查看值

---

## 方法二：使用调试脚本

### 1. 调试完整工作流

```bash
cd /home/data1/gjl/more_learning/shock_invest_Agent/Finance/Financial-MCP-Agent

# 使用调试脚本
./debug_run.sh --command "帮我看看茅台(600519)这只股票值得投资吗"

# 查看日志
tail -f debug_logs/debug_*.log
```

### 2. 测试单个 Agent

```bash
# 测试基本面分析 Agent
python test_agent.py --agent fundamental --stock sh.600519

# 测试技术分析 Agent
python test_agent.py --agent technical --stock sh.600519

# 测试估值分析 Agent
python test_agent.py --agent value --stock sh.600519

# 测试新闻分析 Agent
python test_agent.py --agent news --stock sh.600519

# 测试完整工作流
python test_agent.py --workflow --stock sh.600519
```

---

## 方法三：在代码中添加调试点

### 1. 导入调试工具

在需要调试的文件顶部添加：

```python
from src.utils.debug_tools import debug_trace, enable_debug_mode, print_state
```

### 2. 启用调试模式

在 `main()` 函数开始处：

```python
async def main():
    enable_debug_mode("debug_logs/manual_debug.log")
    # ... 其余代码
```

### 3. 添加调试追踪点

```python
# 追踪变量
debug_trace("检查股票代码", {"stock_code": stock_code, "company_name": company_name})

# 追踪状态
print_state(initial_state, "初始化状态")

# 追踪异常
try:
    result = await some_function()
except Exception as e:
    debug_trace("函数调用失败", {"error": str(e), "traceback": traceback.format_exc()}, "ERROR")
    raise
```

---

## 方法四：使用 Python 调试器 (pdb)

### 在代码中添加断点

```python
import pdb

# 在需要暂停的地方添加
pdb.set_trace()

# 或使用 breakpoint() (Python 3.7+)
breakpoint()
```

### 运行程序

```bash
python src/main.py --command "分析茅台"
```

### pdb 命令

- `n` (next): 执行下一行
- `s` (step): 进入函数
- `c` (continue): 继续执行
- `p variable`: 打印变量
- `pp variable`: 美化打印
- `l`: 显示当前代码
- `w`: 显示调用栈
- `q`: 退出

---

## 关键调试点推荐

### 1. 工作流执行流程

**文件**: `src/main.py`

```python
# 第 115-145 行：工作流定义
workflow.add_node("fundamental_analyst", fundamental_agent)
workflow.add_node("technical_analyst", technical_agent)

# 第 445 行：工作流执行
final_state = await app.ainvoke(initial_state)
```

### 2. Agent 执行流程

**文件**: `src/agents/fundamental_agent.py`

```python
# 第 95-103 行：创建 LLM
llm = create_fixed_chat_openai(...)

# 第 128-131 行：创建 ReAct Agent
agent = create_react_agent(llm, mcp_tools)

# 第 170 行：执行 Agent
response = await agent.ainvoke(input_data)
```

### 3. MCP 工具调用

**文件**: `src/tools/mcp_client.py`

```python
# 第 55 行：获取 MCP 工具
loaded_tools = await _mcp_client_instance.get_tools()
```

### 4. 状态传递

在每个 Agent 的开始和结束处：

```python
# Agent 开始
current_data = state.get("data", {})
print(f"输入数据: {current_data.keys()}")

# Agent 结束
return {"data": current_data, "messages": current_messages, "metadata": current_metadata}
```

---

## 查看日志

### 执行日志

```bash
# 查看最新的执行日志
ls -lt logs/
cat logs/20251117_*/execution_log.json

# 实时查看
tail -f logs/20251117_*/agent_*.log
```

### 调试日志

```bash
# 查看调试日志
tail -f debug_logs/debug_*.log

# 搜索特定内容
grep "ERROR" debug_logs/debug_*.log
grep "tool_calls" debug_logs/debug_*.log
```

---

## 环境变量调试

创建 `.env.debug` 文件：

```bash
# 启用详细日志
DEBUG_MODE=true
TRANSFORMERS_VERBOSITY=info
LANGCHAIN_VERBOSE=true

# 模型配置
OPENAI_COMPATIBLE_MODEL=Qwen/Qwen2.5-72B-Instruct
OPENAI_COMPATIBLE_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_COMPATIBLE_API_KEY=your_api_key

# 禁用代理
http_proxy=
https_proxy=
```

使用：

```bash
# 加载调试配置
set -a
source .env.debug
set +a

python src/main.py --command "测试查询"
```

---

## 性能分析

使用 Python 的 `cProfile`：

```bash
python -m cProfile -o profile_output.prof src/main.py --command "分析茅台"

# 查看结果
python -c "import pstats; p = pstats.Stats('profile_output.prof'); p.sort_stats('cumulative'); p.print_stats(20)"
```

---

## 常见问题调试

### 1. 工具调用失败

在 `src/utils/chat_openai_wrapper.py` 添加日志：

```python
def _patched_convert_dict_to_message(response: Dict[str, Any]) -> AIMessage:
    print(f"DEBUG: Response structure: {json.dumps(response, indent=2, ensure_ascii=False)}")
    # ... 其余代码
```

### 2. Agent 无响应

检查 LangGraph 执行：

```python
# 在 main.py 中
async for event in app.astream(initial_state):
    print(f"Event: {event}")
```

### 3. 数据传递问题

在每个 Agent 开始处：

```python
print(f"Agent input keys: {state.get('data', {}).keys()}")
print(f"Stock code: {state.get('data', {}).get('stock_code')}")
```

---

## 推荐调试工作流

1. **快速定位问题**：使用 `./debug_run.sh` 查看完整日志
2. **精确调试**：在 VS Code 中设置断点，使用 `F5` 启动调试
3. **单元测试**：使用 `test_agent.py` 测试单个组件
4. **性能分析**：使用 `cProfile` 找出性能瓶颈

祝调试顺利！🐛✨
