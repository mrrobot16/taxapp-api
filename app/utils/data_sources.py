from app.config import FLOWS_DIR, IRS_FORMS_DIR

def count_forms_on_disk() -> int:
    if not IRS_FORMS_DIR.is_dir():
        return 0
    return len(list(IRS_FORMS_DIR.glob("*.pdf")))
def count_flow_docs_on_disk(*, skip_empty: bool = False) -> int:
    if not FLOWS_DIR.is_dir():
        return 0
    count = 0
    for flow_dir in sorted(FLOWS_DIR.rglob("*")):
        if not flow_dir.is_dir():
            continue
        for txt_path in sorted(flow_dir.glob("*.txt")):
            if skip_empty and not txt_path.read_text(encoding="utf-8", errors="ignore").strip():
                continue
            count += 1
    return count