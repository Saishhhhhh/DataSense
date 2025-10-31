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
        # Pattern: Number followed by period and space
        pattern = r'(\d+)\.\s+'
        parts = re.split(pattern, text)
        suggestions = []
        
        # After split: parts[0] is text before first match, then alternating: number, content, number, content...
        # Skip parts[0] if it's not empty (text before first numbered item)
        start_idx = 1 if parts[0].strip() == "" else 0
        
        # Process pairs: number at odd indices, content at even indices
        i = start_idx
        while i < len(parts):
            if i + 1 < len(parts):
                # parts[i] might be a number (skip), parts[i+1] is the content
                desc = parts[i + 1].strip() if i + 1 < len(parts) else parts[i].strip() if i < len(parts) else ""
                if desc:
                    # Extract chart type and description
                    # Pattern: "Chart Type: Description"
                    if ':' in desc:
                        potential_split = desc.split(':', 1)
                        if len(potential_split) == 2:
                            chart_type_candidate = potential_split[0].strip()
                            desc_text = potential_split[1].strip()
                            # Check if chart_type_candidate looks like a chart type
                            if any(word in chart_type_candidate for word in ['Chart', 'Bar', 'Line', 'Area', 'Scatter', 'Pie', 'Heatmap']):
                                chart_type = chart_type_candidate
                            else:
                                chart_type = "Visualization"
                                desc_text = desc
                        else:
                            chart_type = "Visualization"
                            desc_text = desc
                    else:
                        # Try to extract from first words
                        words = desc.split()
                        chart_type = "Visualization"
                        desc_text = desc
                        # Look for chart-related keywords in first few words
                        for j, word in enumerate(words[:5]):
                            if word in ['Chart', 'Bar', 'Line', 'Area', 'Scatter', 'Pie', 'Heatmap', 'Box', 'Histogram']:
                                # Take words before and including this keyword
                                chart_type = ' '.join(words[:j+1]) if j+1 < len(words) else 'Visualization'
                                desc_text = ' '.join(words[j+1:]) if j+1 < len(words) else desc
                                break
                    
                    suggestions.append({"type": chart_type, "description": desc_text})
                i += 2  # Skip number and content, move to next number
            else:
                break
        
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


