import json

def get_stage_analysis_json(path: str) -> dict[str, any]:
    """Load stage analysis results from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, any] = json.load(f)
            
        # print(f"stage analysis results: {data}")
        return data
    except FileNotFoundError:
        print(f"Stage analysis results not found: {path}")
        return {}