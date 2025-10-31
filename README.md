<div align="center">

# 📊 DataSense 

### *Ask Data. Get Answers. No Code.*

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
[![LangChain](https://img.shields.io/badge/LangChain-Core-green.svg)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DataSense** is an AI-powered natural-language data analysis assistant that turns plain questions into visual insights.

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
- **Google**: `gemini-2.5-pro`, `gemini-2.5-flash`, etc.
- **Anthropic**: `claude-3-opus`, `claude-3-sonnet`, etc.

---

## 📊 Supported File Formats

- **CSV** (`.csv`) - Comma-separated values
- **Excel** (`.xlsx`, `.xls`) - Microsoft Excel files

**Limitations:**
- Maximum recommended file size: 200MB
- Very large datasets may experience slower processing
- Excel files with multiple sheets will load the first sheet

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📺 Demo Video

Watch the full walkthrough and see DevNotes-AI in action:

**[🎬 Watch Demo Video](https://drive.google.com/file/d/1dtJfPcS9UmK-i_jK_EQFJ_TgMv7N8Ic6/view?usp=drive_link)**

---

<div align="center">

**Made with ❤️ by Saish**

⭐ Star this repo if you find it useful!

</div>