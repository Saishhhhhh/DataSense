# 📊 DataSense - AI-Powered Data Analysis Platform

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Transform your data into insights with AI-powered analysis**

[Features](#-features) • [Installation](#️-installation) • [Usage](#-usage) • [Video Demo](#-video-demo) • [Documentation](#-documentation)

</div>

---

## 🎯 Overview

DataSense is a comprehensive Streamlit-based data analysis platform that empowers users to interact with their data using Natural Language. Upload CSV or Excel files and leverage AI to perform exploratory data analysis, ask questions in plain English, and generate visualizations—all without writing a single line of code.

### Why DataSense?

- 🤖 **AI-Powered Analysis**: Leverage GPT-4, Gemini, or Claude to analyze your data
- 💬 **Natural Language Queries**: Ask questions in plain English, no SQL or Python required
- 📊 **Instant Visualizations**: Generate beautiful charts and plots with simple descriptions
- 🔍 **Comprehensive EDA**: Get detailed statistical insights automatically
- 🔒 **Secure & Safe**: Built-in security guardrails prevent dangerous operations
- 🎨 **Beautiful UI**: Modern, intuitive interface for seamless data exploration

---

## ✨ Features

### 1. 📈 Exploratory Data Analysis (EDA)
Comprehensive automated statistical analysis with beautiful visualizations:

- **Overview Metrics**: Total rows, columns, duplicates, and missing values at a glance
- **Data Preview**: Interactive DataFrame preview with pagination
- **Column Information**: Complete data type, uniqueness, and missing value analysis
- **Missing Values Visualization**: Interactive bar charts showing missing data patterns
- **Numeric Statistics**: Mean, median, quartiles, outliers, and standard deviations
- **Correlation Matrix**: Heatmap visualization of numeric column relationships
- **Categorical Analysis**: Top values with frequency distributions and bar charts
- **Export Reports**: Download complete EDA results as JSON

### 2. 💬 Natural Language Query (NLQ)
Ask questions about your data in plain English:

- **Intelligent Query Processing**: Uses pandas dataframe agent to execute real queries
- **Real Results**: Get actual computed values, not approximations
- **Smart Suggestions**: AI-generated question suggestions based on your data
- **Complex Queries**: Supports grouping, filtering, aggregations, and more
- **Safe Execution**: Code execution in restricted environment

**Example Queries:**
- "What is the average age of customers?"
- "Show me the top 5 countries by total revenue"
- "How has content production changed year-over-year?"
- "What percentage of orders have missing customer information?"

### 3. 📊 AI-Powered Visualization
Generate professional visualizations using natural language:

- **Natural Language Descriptions**: Simply describe what chart you want
- **Auto-Generated Code**: AI writes matplotlib/seaborn code for you
- **Smart Suggestions**: Get visualization recommendations based on your data
- **Multiple Chart Types**: Bar charts, line plots, scatter plots, heatmaps, and more
- **Individual Suggestion Buttons**: One-click chart generation from suggestions
- **Code Preview**: View and learn from generated visualization code

**Example Requests:**
- "Create a bar chart showing top 10 countries by revenue"
- "Plot a line chart of sales trends over time, grouped by product category"
- "Show me a scatter plot of price vs. quantity with colored regions"

### 4. 🔧 Dataframe Manipulator
Transform and clean your data with AI assistance:

- **Natural Language Instructions**: Describe the transformation you need
- **Auto-Generated Pandas Code**: AI writes the code for data manipulation
- **Preview & Download**: See changes before applying, download updated CSV
- **Safe Execution**: Security guardrails prevent dangerous operations
- **Code Transparency**: View generated code to understand transformations

**Example Requests:**
- "Remove rows with missing values in the 'email' column"
- "Keep only columns: name, age, and salary, then sort by salary descending"
- "Group by category and calculate average price for each"

---

## 🛠️ Installation

### Prerequisites

- **Python 3.10+** (required for modern type hints)
- **pip** package manager
- **API Key** from one or more providers:
  - OpenAI API key
  - Google Gemini API key  
  - Anthropic Claude API key

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/DataSense.git
cd DataSense
```

#### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Get API Keys

Choose one or more AI providers and get your API keys:

- **OpenAI**: Get your key from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Google Gemini**: Get your key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Anthropic Claude**: Get your key from [Anthropic Console](https://console.anthropic.com/)

#### 5. Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`

---

## 🎬 Video Demo

<div align="center">

### 📺 Watch DataSense in Action

[![DataSense Demo Video](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

**[🎥 Click here to watch the full demo video](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)**

*Replace `YOUR_VIDEO_ID` with your actual YouTube video ID. You can also use Vimeo, Loom, or any other video platform by updating the embed URL.*

</div>

### Video Highlights

- ⚡ Quick setup and installation
- 📊 EDA feature demonstration
- 💬 Natural Language Query examples
- 📈 Visualization generation walkthrough
- 🔧 Dataframe manipulation showcase

---

## 🎯 Usage Guide

### Getting Started

1. **Upload Your Data**
   - Click "Browse files" in the sidebar
   - Select a CSV or Excel (.xlsx, .xls) file
   - Wait for file processing

2. **Configure AI Settings**
   - Select your preferred AI provider (Gemini, OpenAI, or Claude)
   - Enter your API key in the input field
   - Choose your model (defaults are optimized)

3. **Explore Your Data**

#### EDA Tab
- Automatically displays comprehensive statistics
- Navigate through numeric and categorical insights
- Download complete EDA report as JSON

#### NLQ Tab
- Type your question in natural language
- Click "Run NLQ" to get instant answers
- Use "Generate NLQ Suggestions" for AI-recommended questions
- Click any suggestion to auto-fill and execute

#### Visualization Tab
- Describe the chart you want in the text area
- Click "Generate Chart" to create visualization
- Use "Generate Visualization Suggestions" for recommendations
- Click individual suggestion buttons to use them
- View generated code in the expandable section

#### Dataframe Manipulator Tab
- Enter your transformation request
- Click "Apply Manipulation" to execute
- Preview changes and download updated CSV

---

## 📁 Project Structure

```
DataSense/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # License file
├── favicon.svg                 # Browser favicon (appears in browser tab)
│
├── utils/                      # Modular utilities
│   ├── __init__.py
│   ├── model.py               # AI model initialization
│   ├── io_utils.py            # File loading utilities
│   ├── eda.py                 # EDA computation logic
│   ├── suggestions.py         # AI suggestion generation
│   ├── nlq.py                 # Natural Language Query agent
│   ├── viz.py                 # Visualization generation
│   └── df_manip.py            # Dataframe manipulation
│
└── test_utils/                 # Original standalone scripts
    ├── EDA.py
    ├── NLQ.py
    ├── Visualization.ipynb
    ├── Insight_suggestor.py
    └── dataframe_manipulation.py
```

---

## 🤖 AI Models Supported

| Provider | Default Model | Use Case |
|----------|--------------|----------|
| **Google Gemini** | `gemini-2.5-flash` | Fast, efficient, good for general analysis |
| **OpenAI** | `gpt-4o-mini` | Balanced performance and cost |
| **Anthropic Claude** | `claude-3-5-sonnet-latest` | Best for complex reasoning |

### Switching Models

You can use any compatible model from your chosen provider:
- **OpenAI**: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`, etc.
- **Google**: `gemini-pro`, `gemini-1.5-pro`, etc.
- **Anthropic**: `claude-3-opus`, `claude-3-sonnet`, etc.

---

## 📊 Supported File Formats

- **CSV** (`.csv`) - Comma-separated values
- **Excel** (`.xlsx`, `.xls`) - Microsoft Excel files

**Limitations:**
- Maximum recommended file size: 100MB
- Very large datasets may experience slower processing
- Excel files with multiple sheets will load the first sheet

---

## 🔒 Security Features

DataSense is built with security as a priority:

- ✅ **Safe Code Execution**: Restricted execution environment prevents dangerous operations
- ✅ **Input Validation**: All user inputs are validated before processing
- ✅ **Security Guardrails**: Blacklist of dangerous operations (file I/O, network access, etc.)
- ✅ **No Data Storage**: Your data never leaves your machine (except API calls to LLM providers)
- ✅ **API Key Security**: Keys are stored securely and never exposed in code
- ✅ **Builtin Restrictions**: Dangerous Python builtins are blocked during code execution

---

## 📝 Example Use Cases

### Business Analytics
- Analyze sales data and identify trends
- Compare performance across regions
- Generate executive dashboards

### Data Science
- Quick exploratory data analysis
- Feature engineering assistance
- Statistical summary generation

### Research
- Analyze survey responses
- Visualize experimental results
- Generate publication-ready charts

---

### Development Setup

```bash
# Install development dependencies (if any)
pip install -r requirements-dev.txt  # Optional

# Run tests (if available)
pytest tests/

# Check code style
flake8 app.py utils/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Streamlit** - Amazing framework for building data apps
- **LangChain** - Powerful LLM integration framework
- **AI Providers** - OpenAI, Google, and Anthropic for incredible AI capabilities
- **Open Source Community** - For the amazing libraries and tools

---

## 📧 Support & Contact

- 🐛 **Report Issues**: [GitHub Issues](https://github.com/yourusername/DataSense/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/yourusername/DataSense/discussions)
- 📧 **Email**: [your-email@example.com](mailto:your-email@example.com)

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

---

<div align="center">

**Made with ❤️ by Saish**

[⬆ Back to Top](#-datasense---ai-powered-data-analysis-platform)

</div>
