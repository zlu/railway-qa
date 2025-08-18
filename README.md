# Railway RAG Project

一个支持用户级别自适应的智能检索增强生成（RAG）系统，用于铁路维修手册和文档查询，使用Ollama、ChromaDB和FastAPI构建。

## 概述

本项目实现了一个智能问答系统，能够根据用户专业水平自动调整回答方式，查询铁路文档和门控维护指南：

- **Ollama** 用于本地LLM推理和嵌入
- **ChromaDB** 用于向量存储和相似性搜索
- **FastAPI** 用于REST API端点
- **LangChain** 用于RAG管道编排
- **🤖 自主学习能力** - 自动检测用户级别并调整回答
- **👥 用户级别自适应** - 支持新手和老手两种模式

## 核心特性

- Document ingestion and chunking
- Vector similarity search
- Multiple LLM model support (gemma3, deepseek-r1)
- REST API with FastAPI
- Jupyter notebook for experimentation
- Persistent vector database storage
- **🚀 智能用户级别检测** - 根据问题内容自动判断用户专业水平
- **🎯 自适应回答生成** - 为不同级别用户提供定制化回答
- **🔧 手动级别选择** - 支持手动选择新手/老手模式
- **📚 多文档类型支持** - 铁路文档、门控维护指南、综合搜索
- **💡 自主学习能力** - 系统能够学习和适应用户需求

## 用户级别功能

### 🆕 新手模式
- **适用人群**: 初学者、新员工、需要基础培训的人员
- **回答特点**:
  - 使用简单易懂的语言，避免过多专业术语
  - 提供详细的背景解释和步骤说明
  - 强调安全注意事项
  - 使用具体的例子和类比
  - 分步骤组织答案，便于理解

### 🔧 老手模式
- **适用人群**: 经验丰富的专业人员、技术专家
- **回答特点**:
  - 使用专业术语和技术规格
  - 提供详细的技术参数和系统信息
  - 包含高级故障排除和诊断方法
  - 涉及系统集成和优化建议
  - 提供最佳实践和行业标准

### 🤖 自动检测模式
- **智能分析**: 系统自动分析问题内容，判断用户级别
- **关键词识别**: 识别技术术语、问题复杂度等指标
- **自适应调整**: 根据检测结果自动调整回答风格

## Project Structure

```
railway_rag_project/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── backend.py               # Core RAG backend logic with user level support
├── fastapi_app.py           # FastAPI web server with level adaptation
├── simple_ui.html           # Simple web UI with level buttons
├── ollama_embeddings.py     # Custom Ollama embeddings wrapper
├── example.ipynb            # Jupyter notebook for testing
├── railways.pdf             # Source railway documentation
├── data/
│   └── Door control units maintenance guideline.pdf  # Door control maintenance guide
├── railway_chroma_db/       # ChromaDB vector database
├── guide.md                 # User guide and FAQ
└── integration.md           # API integration examples
```

## Prerequisites

1. **Ollama** installed and running locally
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service
   ollama serve
   ```

2. **Required models** pulled in Ollama:
   ```bash
   # LLM models
   ollama pull gemma3
   ollama pull deepseek-r1
   
   # Embedding model
   ollama pull nomic-embed-text
   ```

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd railway_rag_project
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Process the documents:**
   ```bash
   python process_door_control.py
   ```

## Usage

### 1. Web界面使用

启动服务器后，访问 `http://localhost:8000` 查看简单Web界面：

```bash
# Start the server
uvicorn fastapi_app:app --reload

# Open simple_ui.html in browser
open simple_ui.html
```

Web界面功能：
- 🆕 新手模式按钮
- 🔧 老手模式按钮  
- 🤖 自动检测按钮
- 示例问题快速选择
- 实时级别显示

### 2. FastAPI Web Server

```bash
# Start the server
uvicorn fastapi_app:app --reload
```

The server will be available at `http://localhost:8000`

#### API Endpoints

- **POST** `/ask` - 智能问答（支持用户级别）
  ```json
  {
    "question": "门控单元的基本维护步骤是什么？",
    "document_type": "door_control",
    "user_level": "beginner",  // "beginner", "expert", or null for auto-detection
    "llm_model": "gemma3"
  }
  ```

- **GET** `/docs` - Interactive API documentation (Swagger UI)
- **GET** `/` - API information and available options
- **GET** `/user-levels` - 获取用户级别信息
- **GET** `/health` - Health check endpoint

#### Example API Usage

```bash
# 新手模式查询
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "门控单元的基本维护步骤是什么？", "document_type": "door_control", "user_level": "beginner"}'

# 老手模式查询
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "门控系统的技术参数和高级故障排除方法有哪些？", "document_type": "door_control", "user_level": "expert"}'

# 自动检测模式
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "门控单元的基本维护步骤是什么？", "document_type": "door_control"}'

# Using Python
import requests
response = requests.post("http://localhost:8000/ask", 
                        json={"question": "门控单元的基本维护步骤是什么？", "document_type": "door_control", "user_level": "beginner"})
print(response.json())
```

### 3. Direct Python Usage

```python
from backend import answer_question

# 新手模式
answer = answer_question("门控单元的基本维护步骤是什么？", document_type="door_control", user_level="beginner", llm_model="gemma3")
print(answer)

# 老手模式
answer = answer_question("门控系统的技术参数和高级故障排除方法有哪些？", document_type="door_control", user_level="expert", llm_model="gemma3")
print(answer)

# 自动检测
answer = answer_question("门控单元的基本维护步骤是什么？", document_type="door_control")
print(answer)
```

## Example Questions and Answers

### 新手模式示例

**Q: 门控单元的基本维护步骤是什么？**

**A:** 好的，让我们来一起学习如何进行门控系统的基本维护，我会用最简单易懂的方式，确保你完全理解每个步骤。

**门控系统维护指南（新手版）**

门控系统是火车车厢门自动开关的关键部件，维护它的目的就是确保门能够安全、可靠地开关。这个过程可以比作你维护自己的电脑，定期清理垃圾、检查硬件，以保证电脑正常运行。

**第一步：安全第一！**

* **停电操作：** 在进行任何维护之前，务必确认火车已经完全停电，并且所有相关电源都已断开。这是最重要的一步，保护你的安全！
* **佩戴防护装备：** 确保你穿着合适的工具和安全服。
* **了解风险：** 熟悉门控系统的结构，了解可能存在的危险，比如高压电、机械伤害等。

**第二步：初步检查 - 观察和评估**

1. **视觉检查：**
   * **门板和轨道：** 仔细观察门板和轨道（也就是门运送的轨道）是否有变形、划痕、异物或腐蚀。就像检查你的自行车车轮是否平整一样，确保没有明显的问题。
   * **滚轮和导轨：** 检查门运送的滚轮和导轨是否磨损、松动或损坏。这些是帮助门顺畅移动的关键部件。
   * **密封条：** 检查门周围的密封条是否完整、没有破损或变形。它们的作用是防止漏风和漏雨。

2. **门板运行状态：** 观察门在自动开关时是否顺畅，是否有异响、卡顿或停滞的情况。

**第三步：关键部件维护（涉及一些操作，需谨慎）**

1. **滚轮和导轨的维护：**
   * **清洁：** 使用干净的布或压缩空气清除滚轮和导轨上的灰尘和污垢。
   * **润滑：** 使用适当的润滑剂（通常是轻薄的油）在滚轮和导轨上涂抹。注意不要过度润滑，以免造成污染。
   * **调整：** 如果发现滚轮位置不正或导轨松动，需要用专门的工具进行调整或拧紧。

2. **密封条的维护：**
   * **检查：** 确认密封条没有损坏或变形。
   * **调整：** 如果密封条松动，需要重新安装，确保其紧密贴合门框。

3. **检查并清理滚轮/传感器** 检查传感器是否被异物阻挡，例如清洁传感器及其周围区域。

**第四步：部分部件的检查与调整**

1. **定位开关：** 检查定位开关是否准确，当门完全关闭时，它必须能够正确指示。
2. **限位开关：** 检查限位开关是否在正确的位置，它们会控制门运动的范围，防止门损坏。
3. **电子元件：** 检查连接的电线和插头，确保它们没有腐蚀或损坏。

**第五步：测试与确认**

1. **模拟开关：** 在确保安全的情况下，对门控系统进行模拟开关测试，以确认所有部件都能正常工作。
2. **观察：** 仔细观察门在开关时是否顺畅、安全，没有出现任何异常情况。

**重要提示和注意事项：**

* **不要强行操作：** 如果遇到无法解决的问题，不要强行操作，应立即寻求专业人士的帮助。
* **记录：** 记录你所做的维护工作，包括更换的部件、调整的参数等。
* **定期维护：** 按照规定的时间间隔进行维护，以确保门控系统的正常运行。

**记住，安全第一！如果你不确定如何进行维护，请务必寻求专业人士的帮助。**

### 老手模式示例

**Q: 门控系统的技术参数和高级故障排除方法有哪些？**

**A:** 好的，老手，针对您提出的"门控系统技术参数和高级故障排除方法"的问题，我将结合文档内容，提供一份全面且专业的解答，务必以技术细节和高级信息为主，并考虑到您作为经验丰富的专业人员的水平。

**门控系统技术参数及高级故障排除方法**

门控系统，尤其是用于双滑式车门（Double Glider Door System）的系统，是一个复杂多样的集成，涉及到机械、电气和控制等多个层面。我们将其分解为关键技术参数和高级故障排除策略。

**一、关键技术参数**

* **DCU (Door Control Unit) 性能指标:**
  * **响应时间:** DCU控制一个车门完全打开或关闭所需的时间，通常小于 3 秒，具体数值需根据门型和负载情况进行优化。
  * **控制精度:** DCU控制门开度及速度的精度，一般为 ±1%，通过精确的电机控制和传感器反馈实现。
  * **电源需求:** DCU通常使用 24V DC 电源，稳压精度对系统稳定至关重要，需控制在 ±5%以内。
  * **逻辑处理能力:** DCU需具备处理各种安全逻辑、状态监测、错误码生成等功能，需满足一定的计算速度和内存要求。

* **电机参数:**
  * **额定扭矩 (Nm):** 电机扭矩直接影响车门的启停速度和承载能力，需根据门型和负载选择合适的电机。
  * **速度 (RPM):** 电机转速影响车门启闭速度，需与门型和系统要求相匹配。
  * **功率 (kW):** 电机功率影响启动和运行时的能耗，需根据负载情况进行优化。

* **传感器：**
  * **Reed Switch：** 用于检测门的状态（开/关），精度要求高，误触发率要低。
  * **安全边缘传感器：** 通常使用 8kΩ 或 1kΩ 的电阻，用于安全检测，其响应速度至关重要。
  * **光幕传感器：** 用于检测车门周围是否有障碍物，响应速度及精度直接关系到系统的安全性能。

* **Air Pressure System (如果采用气动)：**
  * **压力 (bar/psi):** 气压对车门启闭速度和力矩有很大影响，需进行精确控制。
  * **流量 (L/min):** 气流量需根据系统需求进行调整，以保证系统稳定运行。

**二、高级故障排除和诊断方法**

* **GfA-Stick & App (诊断工具):** 利用 GfA-Stick 及配套 App，可以实时读取 DCU 的故障日志、参数设置及事件数据。通过 Bluetooth 连接，可以远程分析故障原因，并进行配置调整。

* **DCU 状态监测:**
  * **实时监控：** 持续监控 DCU 的运行状态，包括电机温度、传感器状态、通信状态等。
  * **参数数据分析：** 分析 DCU 的参数设置，例如电机扭矩、速度、步进进程等，找出与系统性能不符的地方。

* **系统联动分析:**
  * **DCU-Sensor 联动：** 检查 DCU 与传感器之间的通信状态，确保传感器数据能够正确地传输到 DCU。
  * **DCU-Motor 联动：** 检查 DCU 与电机的控制状态，确保电机能够按照 DCU 的指令进行启动和停止。

* **故障诊断流程:**
  * **症状识别：** 仔细分析故障现象，例如门不响应、门速度异常、门卡住等。
  * **初步判断：** 根据故障现象，初步判断故障原因，例如 DCU 故障、电机故障、传感器故障、空气压力不足等。
  * **逐一排查：** 利用上述工具和方法，逐一排查各个部件的故障原因。
  * **系统复位：** 在进行复杂故障排除前，通常需要先进行 DCU 的复位，以清除错误状态。

**三、系统集成与优化建议**

* **传感器数据集成：** 充分利用安全边缘传感器和光幕传感器的数据，可以实现更精确的门控系统控制，提高系统的安全性。
* **电机控制优化：** 采用先进的电机控制算法，可以提高电机的控制精度，减少能量损耗，延长电机的使用寿命。
* **系统集成测试：** 在系统集成完成后，进行全面的功能测试和性能测试，确保系统能够稳定可靠地运行。

**四、最佳实践和行业标准**

* 遵循 LUL 的 1996 库存缺陷指南和缺陷在服务中的说明。
* 定期维护和检查门控系统，及时更换易损件。
* 进行风险评估，识别潜在的故障风险，并采取相应的预防措施。
* 进行定期培训，提高工作人员的专业技能和安全意识。

**五、相关技术规范和标准**

* **ISO 13849-1:** 功能安全标准，对安全关键系统进行设计和验证。
* **EN 61508:** 功能安全标准，对安全相关的电子电气系统进行设计和验证。

### Railway Documentation

**Q: What are the 4 steps of the TAR?**

**A:** The Technical Assessment Report (TAR) was delivered in four steps:

1. **ORR prepared a Request For Information (RFI)** and gathered relevant documents
2. **Meetings with Regional representatives** to clarify the provided documents
3. **Interviews with a Head of Maintenance/Infrastructure Director** from each Region to identify alignment and key observations
4. **ORR compiled, reviewed, and identified observations and conclusions** from the collected information for the report

**Q: What is the scope of this assessment?**

**A:** The scope of this assessment is to **interpret and assess Network Rail's (NR) current approach to maintenance**. It aims to:
- Inform ORR's understanding of current NR practices for use in its business-as-usual monitoring against the Network Licence
- Contribute to the considerations for Periodic Review 2023 (PR23)

### Combined Search

**Q: What are the maintenance procedures for railway equipment?**

**A:** Maintenance procedures include:

1. **General Maintenance Philosophy:**
   - Planning and control activities to manage risks and opportunities
   - Continuous improvement culture for safety and asset performance

2. **Door Control Maintenance:**
   - Routine downloading and analysis of train management system data
   - Operator awareness training for sleep and nutrition effects
   - Fault handling protocols with safety briefings and notifications

3. **Specific Equipment Maintenance:**
   - Control unit inspection, testing, software updates, and replacement
   - Door cut-out testing and repair procedures
   - Circuit integrity and access control measures

4. **Safety and Documentation:**
   - Isolation of malfunctioning doors using door isolation locks
   - Record keeping in defect tracking systems
   - Training crews to identify and isolate faults

## Configuration

Edit `backend.py` to customize:
- **Embedding model**: `EMBED_MODEL_NAME = "nomic-embed-text"`
- **Default LLM**: `DEFAULT_LLM_MODEL = "gemma3"`
- **Vector database location**: `PERSIST_DIRECTORY = "railway_chroma_db"`
- **Collection names**: 
  - `RAILWAY_COLLECTION = "railway_document_embeddings"`
  - `DOOR_CONTROL_COLLECTION = "door_control_embeddings"`

### 用户级别配置

在 `backend.py` 中可以自定义用户级别特征：

```python
USER_LEVELS = {
    "beginner": {
        "name": "新手",
        "description": "初学者，需要基础解释和详细步骤",
        "characteristics": [
            "需要详细的背景解释",
            "喜欢分步骤的指导",
            "需要安全提醒",
            "偏好简单易懂的语言",
            "需要更多示例和类比"
        ]
    },
    "expert": {
        "name": "老手", 
        "description": "经验丰富的专业人员，需要技术细节和高级信息",
        "characteristics": [
            "需要技术规格和参数",
            "偏好专业术语",
            "关注高级故障排除",
            "需要系统集成信息",
            "关注最佳实践和优化"
        ]
    }
}
```

### Memory Optimization

For low-RAM systems (8GB or less):
- Increase `chunk_size` in the text splitter (e.g., 3000 instead of 2000)
- Reduce `chunk_overlap` 
- Process documents in smaller batches

## Troubleshooting

### Common Issues

1. **Ollama not running**: Make sure `ollama serve` is running
2. **Model not found**: Pull required models with `ollama pull <model_name>`
3. **Memory issues**: See the memory optimization section above
4. **ChromaDB errors**: Don't manually modify files in `railway_chroma_db/`

### Memory Usage

The system loads all embeddings into RAM. For large documents:
- Use larger chunk sizes
- Consider using FAISS instead of ChromaDB
- Process documents in batches

## Development

### Adding New Documents

1. Place new PDFs in the `data/` directory
2. Update the document processing script or create a new one
3. Re-run the embedding creation process

### Customizing the RAG Pipeline

- Modify `backend.py` to change retrieval parameters
- Adjust chunk size and overlap in the text splitter
- Change the number of retrieved documents (`k` parameter)

### 扩展用户级别

要添加新的用户级别，可以修改 `backend.py` 中的 `USER_LEVELS` 和 `LEARNING_PATTERNS` 配置：

```python
# 添加中级用户级别
"intermediate": {
    "name": "中级",
    "description": "有一定经验的操作人员",
    "characteristics": [
        "需要平衡的详细程度",
        "偏好实用指导",
        "关注操作要点"
    ]
}
```

## License

This project is part of the Wiseman system. See the main project for licensing information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request
