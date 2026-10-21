import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "evaluation"

MUST_ASK = {
    "dev-003": [
        "Can you provide the exact Streamlit version and a minimal reproduction that shows the behavior with and without st.session_state, including whether the problem still occurs on the latest Streamlit release?"
    ],
    "dev-004": [
        "Which exact _stcore endpoint, HTTP response, and response headers/body triggered each remaining disclosure finding, and which Streamlit version was tested for each finding?",
        "Can you provide the sanitized DAST request/response evidence for the remaining findings so we can distinguish framework behavior from configuration or proxy behavior?"
    ],
    "dev-005": [
        "Does the interaction failure occur with both mouse and keyboard input, and does it reproduce with a minimal st.dialog containing only st.color_picker?",
        "Which browser versions and Streamlit version reproduce the issue, and does the same minimal example work when st.color_picker is placed in st.popover or outside an overlay?"
    ],
    "dev-008": [
        "What exact Streamlit version, Python version, operating system, browser, and deployment environment produced the Cached ForwardMsg MISS?",
        "Can you provide a minimal reproducible application and indicate whether the error occurs after a specific rerun, reconnect, or widget interaction?"
    ],
    "dev-009": [
        "Does the warning appear only during the initial page load on slow connections, and does it disappear after the form finishes rendering?",
        "Can you provide the approximate network conditions and confirm whether the issue reproduces with the official minimal form example on the same Streamlit version?"
    ],
    "dev-010": [
        "Can you provide a minimal reproduction containing only st.tabs and the smallest widget/session-state interaction sequence that triggers the collapse?",
        "Does the failure reproduce consistently on 1.59.0 and disappear on 1.58.x with the same code, Python version, and browser?"
    ],
    "dev-012": [
        "Can you provide the exact Starlette and Uvicorn versions installed alongside Streamlit 1.61.0 and confirm whether upgrading specifically to Streamlit 1.61.1 resolves the startup error?",
        "Does the same failure reproduce outside conda-forge, for example in a clean pip environment with the same Python version?"
    ],
    "dev-014": [
        "Can you provide the exact DataFrame schema and representative values that contain the mixed list/non-list structure, plus the complete traceback?",
        "Does converting the nested structure to a uniform Arrow-compatible type change the behavior, and which Streamlit/PyArrow versions are involved?"
    ],
    "test-001": [
        "What exact websocket close/reconnect sequence causes the existing session to be lost, and can you provide logs showing the session identifier before and after the reconnect?",
        "Does the behavior reproduce consistently under a minimal application with a controlled network interruption, and which Streamlit version and browser are involved?"
    ],
    "test-002": [
        "Can you provide the exact websockets and Uvicorn versions, the reverse-proxy/authentication setup, and the specific non-ASCII header value that triggers the failure?",
        "Does the WebSocket handshake succeed when the same deployment uses an ASCII-only header value or websockets<16.1?"
    ],
    "test-003": [
        "Can you provide the exact resolved dependency versions from the failing environment and confirm whether the same GZipResponder error occurs in a clean non-Community-Cloud environment?",
        "Does pinning a compatible Starlette version or upgrading Streamlit to the reported fixed version eliminate the startup error?"
    ],
    "test-004": [
        "Can you provide a minimal reproducible application, the exact Streamlit version, and the client/server logs around the Cached ForwardMsg MISS?",
        "Does the error occur after a reconnect or rerun, and can you determine whether the cached message hash is missing from the client cache at the time of failure?"
    ],
    "test-006": [
        "Can you provide the minimal widget code and exact sequence of hiding and showing the keyed widget that causes the UI value and st.session_state value to diverge?",
        "Which Streamlit commit/version is being tested, and does the mismatch occur on the first remount run before a subsequent rerun overwrites the session state?"
    ],
    "test-009": [
        "Can you provide the smallest reproduction with an st.tabs widget and one interactive widget inside a tab that triggers the failure?",
        "Does the regression reproduce consistently on 1.59.0 and disappear on 1.58.0 with the same browser and Python environment?"
    ],
    "test-010": [
        "Can you provide the exact Streamlit, Python, Pandas, and relevant dependency versions and confirm whether the failure requires the output of one cached function to be passed into another cached function?",
        "Does the same DataFrame hash successfully when the nested list is removed or when the intermediate cached result is not passed to the second cached function?"
    ],
    "test-011": [
        "Can you provide the minimal chat_input code and the exact sequence of state updates/callbacks used while the model response is generated?",
        "Does moving the update into an on_submit callback or using st.rerun change the behavior, and which Streamlit version is being tested?"
    ],
    "test-012": [
        "Can you provide the minimal data editor configuration and fragment code, including the run_every interval and the SelectboxColumn options that cannot be selected?",
        "Does the problem disappear when the fragment's run_every is disabled, and does it reproduce without st.session_state as reported?"
    ],
    "test-013": [
        "Can you provide the minimal AppTest reproduction and the exact widget-removal sequence that causes the 'We'll never get here' condition?",
        "Does the error occur only when input widgets are removed between reruns, and which Streamlit/AppTest version or commit is being tested?"
    ],
}

def load_cases(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data, data.get("cases", [])
    return data, data

def save_cases(path, original, cases):
    if isinstance(original, dict):
        original["cases"] = cases
        output = original
    else:
        output = cases
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

for filename in ("dev.json", "test.json"):
    path = EVAL_DIR / filename
    original, cases = load_cases(path)
    changed = 0

    for case in cases:
        case_id = case["case_id"]
        if case_id not in MUST_ASK:
            continue

        target_turn = case["turns"][-1]
        target_turn["must_ask"] = MUST_ASK[case_id]
        changed += 1

    save_cases(path, original, cases)
    print(f"{filename}: updated {changed} cases")

print("Total updated:", len(MUST_ASK))
print("Next: python scripts\\validate_evaluation.py")
print("Then: python scripts\\review_evaluation_quality.py")
