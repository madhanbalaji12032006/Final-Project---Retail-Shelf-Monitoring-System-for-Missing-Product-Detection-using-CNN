import json
from pathlib import Path

def load_categories_from_coco(annotations_path):
    """
    Returns:
        cat_id_to_name: dict[int, str]
        name_to_cat_id: dict[str, int]
        class_names: list[str] in category id order (sorted by id)
    """
    annotations_path = Path(annotations_path)
    with annotations_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cats = data.get("categories", [])
    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}
    class_names = [cat_id_to_name[k] for k in sorted(cat_id_to_name.keys())]
    return cat_id_to_name, name_to_cat_id, class_names
