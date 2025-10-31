from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

def get_dataframe_details(df, n_rows: int = 5) -> str:
    return f"""
Columns: {', '.join(df.columns.tolist())}

Data Types:
{df.dtypes.to_string()}

Shape: {df.shape[0]} rows × {df.shape[1]} columns

Sample Data:
{df.head(n_rows).to_string(index=False)}
""".strip()

def get_insight_suggestions(model, df):
    dataframe_details = get_dataframe_details(df)
    response_schemas = [
        ResponseSchema(name="analytical_questions", description="List of 5 insightful natural language queries for analysis."),
        ResponseSchema(name="visualization_suggestions", description="List of 5 visualizations with chart types and columns to plot."),
    ]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()
    template = PromptTemplate(
        template="""
You are a skilled Python data analyst and EDA expert.

Your job is to carefully study the given dataframe details and suggest useful **analytical questions** and **visualizations** 
that can help a data analyst gain deeper insights into this dataset.

### DataFrame Details:
{dataframe_details}

### Output Format:
{format_instructions}
""",
        input_variables=["dataframe_details", "format_instructions"],
    )
    chain = template | model | parser
    result = chain.invoke({"dataframe_details": dataframe_details, "format_instructions": format_instructions})
    return result

def normalize_nlq_suggestions(suggestions_obj) -> list:
    raw = suggestions_obj or {}
    items = raw.get("analytical_questions", []) if isinstance(raw, dict) else raw
    out = []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, str):
                s = it.strip()
                if s:
                    out.append(s)
            elif isinstance(it, dict):
                val = it.get("question") or it.get("text") or it.get("value")
                if isinstance(val, str) and val.strip():
                    out.append(val.strip())
    elif isinstance(items, str):
        for line in items.splitlines():
            s = line.strip(" -•\t\r\n")
            if len(s) > 1:
                out.append(s)
    dedup = []
    seen = set()
    for s in out:
        if s and s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup[:5]

def normalize_viz_suggestions(suggestions_obj) -> list:
    raw = suggestions_obj or {}
    items = raw.get("visualization_suggestions", []) if isinstance(raw, dict) else raw
    normalized = []
    
    def split_numbered_suggestions(text: str) -> list:
        """Split text like '1. ... 2. ... 3. ...' into individual suggestions"""
        import re
        suggestions = []
        
        # Enhanced pattern to handle: "1. **Title:** Description. 2. **Title:** Description."
        # Match: number, period, optional space, optional bold markers, title, colon, description
        # Pattern captures: number, optional bold title, and description
        pattern = r'(\d+)\.\s*(?:\*\*)?([^*:]+?)(?:\*\*)?:\s*(.+?)(?=\d+\.\s*(?:\*\*)?|$)'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            num = match.group(1)
            title_part = match.group(2).strip()
            desc = match.group(3).strip()
            
            # Clean up title (remove bold markers, extra spaces)
            title_part = re.sub(r'\*\*', '', title_part).strip()
            
            # Extract chart type from title or description
            chart_type = "Visualization"
            desc_text = desc
            
            # Check if title contains chart type keywords
            title_lower = title_part.lower()
            chart_keywords = {
                'dashboard': 'Dashboard',
                'bar chart': 'Bar Chart',
                'boxplot': 'Boxplot',
                'stacked bar': 'Stacked Bar Chart',
                'line plot': 'Line Chart',
                'line chart': 'Line Chart',
                'scatter': 'Scatter Plot',
                'heatmap': 'Heatmap',
                'histogram': 'Histogram'
            }
            
            for keyword, chart_name in chart_keywords.items():
                if keyword in title_lower:
                    chart_type = chart_name
                    break
            
            # Also check description for chart types
            if chart_type == "Visualization":
                desc_lower = desc.lower()
                # Look for patterns like "bar chart", "line plot", etc.
                for keyword, chart_name in chart_keywords.items():
                    if keyword in desc_lower:
                        chart_type = chart_name
                        break
                
                # Check for standalone chart words
                words = desc.split()
                for j, word in enumerate(words[:8]):  # Check first 8 words
                    word_lower = word.lower().rstrip('s')  # Remove plural
                    if word_lower in ['chart', 'plot', 'graph'] and j > 0:
                        # Get preceding words for context
                        context = ' '.join(words[max(0, j-2):j+1])
                        if 'bar' in context.lower():
                            chart_type = 'Bar Chart'
                        elif 'line' in context.lower():
                            chart_type = 'Line Chart'
                        elif 'scatter' in context.lower():
                            chart_type = 'Scatter Plot'
                        break
            
            # Use title as part of description if meaningful
            if title_part and len(title_part) > 3:
                if title_part not in desc_text:
                    desc_text = f"{title_part}: {desc_text}"
            
            suggestions.append({
                "type": chart_type,
                "description": desc_text.rstrip('. '),  # Remove trailing periods/spaces
                "original_title": title_part
            })
        
        # Fallback: if no matches found, try simpler pattern
        if not suggestions:
            # Try pattern without bold markers
            simple_pattern = r'(\d+)\.\s+([^:]+?):\s*(.+?)(?=\d+\.|$)'
            simple_matches = re.finditer(simple_pattern, text, re.DOTALL)
            for match in simple_matches:
                title_part = match.group(2).strip()
                desc = match.group(3).strip()
                chart_type = "Visualization"
                desc_text = desc
                
                # Extract chart type
                if any(word in title_part.lower() for word in ['chart', 'plot', 'graph', 'dashboard']):
                    words = title_part.split()
                    for word in words:
                        if word.lower() in ['bar', 'line', 'scatter', 'box', 'pie']:
                            chart_type = word.capitalize() + ' Chart'
                            break
                
                suggestions.append({
                    "type": chart_type,
                    "description": desc_text.rstrip('. '),
                    "original_title": title_part
                })
        
        return suggestions if suggestions else [{"type": "Visualization", "description": text}]
    
    def to_prompt(obj):
        if not obj:
            return None
        if isinstance(obj, str):
            return obj.strip()
        if isinstance(obj, dict):
            t = obj.get("type")
            desc = obj.get("description") or obj.get("desc") or obj.get("text")
            cols = obj.get("columns") or obj.get("cols") or []
            cols_txt = ", ".join([str(c) for c in cols]) if cols else "relevant columns"
            if t and desc:
                return f"Create a {t} — {desc} Use columns {cols_txt}."
            if desc:
                return f"{desc}"
            return None
        return None

    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                # Check if description contains numbered suggestions
                desc = it.get("description") or it.get("desc") or it.get("text") or ""
                if desc and (desc.count(".") > 2 and any(char.isdigit() for char in desc[:50])):
                    # Likely contains numbered suggestions - split them
                    split_items = split_numbered_suggestions(desc)
                    for split_item in split_items:
                        prompt = to_prompt(split_item)
                        normalized.append({
                            "type": split_item.get("type", it.get("type", "Visualization")),
                            "description": split_item.get("description", ""),
                            "original_title": split_item.get("original_title", ""),
                            "columns": it.get("columns") or it.get("cols") or [],
                            "prompt": prompt or "",
                        })
                else:
                    # Normal dict item
                    prompt = to_prompt(it)
                    normalized.append({
                        "type": it.get("type") or "Visualization",
                        "description": it.get("description") or it.get("desc") or it.get("text") or "",
                        "columns": it.get("columns") or it.get("cols") or [],
                        "prompt": prompt or "",
                    })
            else:
                s = str(it).strip()
                if s:
                    # Check if it's a numbered list
                    if s.count(".") > 2 and any(char.isdigit() for char in s[:50]):
                        split_items = split_numbered_suggestions(s)
                        for split_item in split_items:
                            normalized.append({
                                "type": split_item.get("type", "Visualization"),
                                "description": split_item.get("description", ""),
                                "original_title": split_item.get("original_title", ""),
                                "columns": [],
                                "prompt": split_item.get("description", ""),
                            })
                    else:
                        normalized.append({
                            "type": "Visualization",
                            "description": s,
                            "columns": [],
                            "prompt": s,
                        })
    elif isinstance(items, str):
        s = items.strip()
        if s:
            # Check if it's a numbered list
            if s.count(".") > 2 and any(char.isdigit() for char in s[:50]):
                split_items = split_numbered_suggestions(s)
                for split_item in split_items:
                    normalized.append({
                        "type": split_item.get("type", "Visualization"),
                        "description": split_item.get("description", ""),
                        "original_title": split_item.get("original_title", ""),
                        "columns": [],
                        "prompt": split_item.get("description", ""),
                    })
            else:
                normalized.append({
                    "type": "Visualization",
                    "description": s,
                    "columns": [],
                    "prompt": s,
                })

    out = []
    seen_prompts = set()
    for obj in normalized:
        p = obj.get("prompt", "").strip()
        if p and p not in seen_prompts:
            seen_prompts.add(p)
            out.append(obj)
    return out[:5]


