"""
Polymer SMILES Descriptor Generator - Frontend

Streamlit application that communicates with the backend API.
"""
import streamlit as st
import pandas as pd
import requests
import io
from typing import Optional

# Configuration
API_BASE_URL = st.sidebar.text_input("API URL", value="http://localhost:8000", key="api_url")

APP_NAME = "Polymer SMILES Descriptor Generator"
APP_VERSION = "1.0.0"

# Page configuration
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "input"
if "parse_result" not in st.session_state:
    st.session_state.parse_result = None
if "results" not in st.session_state:
    st.session_state.results = {}
if "providers" not in st.session_state:
    st.session_state.providers = None


def api_request(method: str, endpoint: str, json: Optional[dict] = None) -> Optional[dict]:
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=30)
        else:
            response = requests.post(url, json=json, timeout=120)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_BASE_URL}. Is the backend running?")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ API request timed out")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API error: {e}")
        return None


def fetch_providers():
    """Fetch providers from API"""
    if st.session_state.providers is None:
        data = api_request("GET", "/api/providers")
        if data:
            st.session_state.providers = data["providers"]
    return st.session_state.providers


def render_sidebar():
    """Render sidebar with navigation and info"""
    with st.sidebar:
        st.title("🧬 " + APP_NAME)
        st.caption(f"v{APP_VERSION}")
        
        st.divider()
        
        # Navigation
        st.subheader("📍 Navigation")
        pages = {
            "input": "1️⃣ Input SMILES",
            "models": "2️⃣ Select Models",
            "results": "3️⃣ View Results"
        }
        
        for page_key, page_name in pages.items():
            if st.button(
                page_name, 
                key=f"nav_{page_key}",
                use_container_width=True,
                type="primary" if st.session_state.page == page_key else "secondary"
            ):
                st.session_state.page = page_key
                st.rerun()
        
        st.divider()
        
        # Status
        if st.session_state.parse_result:
            result = st.session_state.parse_result
            st.subheader("📊 Input Status")
            st.metric("Total", result.get("total_count", 0))
            col1, col2 = st.columns(2)
            col1.metric("✅ Valid", result.get("success_count", 0))
            col2.metric("❌ Error", result.get("error_count", 0))
        
        st.divider()
        
        # API Status
        st.subheader("🔌 API Status")
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                st.success("Connected")
            else:
                st.error("Disconnected")
        except:
            st.error("Disconnected")


def render_input_page():
    """Render SMILES input page"""
    st.header("1️⃣ Input Polymer SMILES")
    
    st.markdown("""
    Enter polymer SMILES or copolymer compositions.
    Polymer wildcards (`*`) are supported.
    """)
    
    # Input tabs
    tab1, tab2 = st.tabs(["📝 Text Input", "📁 File Upload"])
    
    with tab1:
        smiles_input = st.text_area(
            "SMILES Input",
            height=200,
            placeholder="CCO\nc1ccccc1\nmol_001,*CC(C)(C)C*",
            help="One SMILES per line, or CSV format: id,smiles",
            key="smiles_text_input"
        )
    
    with tab2:
        uploaded_file = st.file_uploader(
            "Upload CSV or TXT file",
            type=["csv", "txt"],
            help="CSV with 'smiles' column, or TXT with one SMILES per line"
        )
        if uploaded_file:
            content = uploaded_file.read().decode("utf-8")
            smiles_input = content
            st.success(f"Loaded {len(content.splitlines())} lines from file")
    
    # Settings
    st.subheader("⚙️ Parsing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        canonicalize = st.checkbox(
            "Canonicalize SMILES",
            value=True,
            help="Convert to canonical form using RDKit"
        )
    
    with col2:
        wildcard_mode = st.selectbox(
            "Polymer Wildcard (*) Handling",
            options=["replace", "skip", "error"],
            index=0,
            help="replace: Convert * to [*]\nskip: Keep as-is\nerror: Raise error"
        )
    
    # Parse button
    if st.button("🔍 Parse & Validate", type="primary", use_container_width=True):
        if not smiles_input or not smiles_input.strip():
            st.error("Please enter at least one SMILES string")
        else:
            with st.spinner("Parsing SMILES via API..."):
                result = api_request("POST", "/api/parse", {
                    "smiles_text": smiles_input,
                    "canonicalize": canonicalize,
                    "wildcard_mode": wildcard_mode
                })
                
                if result:
                    st.session_state.parse_result = result
    
    # Show results
    if st.session_state.parse_result:
        result = st.session_state.parse_result
        
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Entries", result["total_count"])
        col2.metric("Successfully Parsed", result["success_count"])
        col3.metric("Errors", result["error_count"])
        
        # Show results table
        if result.get("records"):
            st.subheader("📋 Parse Results")
            
            df_data = []
            for r in result["records"]:
                df_data.append({
                    "ID": r["input_id"],
                    "Input SMILES": r["input_smiles_raw"][:50] + "..." if len(r["input_smiles_raw"]) > 50 else r["input_smiles_raw"],
                    "Status": "✅" if r["parse_status"] == "OK" else "❌",
                    "Polymer (*)": "Yes" if r["has_polymer_wildcard"] else "",
                    "Error": r.get("error_message", "")[:30] if r.get("error_message") else ""
                })
            
            st.dataframe(pd.DataFrame(df_data), use_container_width=True)
        
        if result["success_count"] > 0:
            st.success(f"✅ {result['success_count']} SMILES ready for descriptor generation")
            if st.button("➡️ Continue to Model Selection", type="primary", key="continue_to_models"):
                st.session_state.page = "models"
                st.rerun()


def render_model_page():
    """Render model selection page"""
    st.header("2️⃣ Select Descriptor Models")
    
    if not st.session_state.parse_result:
        st.warning("Please parse SMILES first")
        if st.button("← Go to Input"):
            st.session_state.page = "input"
            st.rerun()
        return
    
    result = st.session_state.parse_result
    st.info(f"📊 {result['success_count']} SMILES entries ready for processing")
    
    # Fetch providers
    providers = fetch_providers()
    if not providers:
        st.error("Cannot fetch providers from API")
        return
    
    # Group by kind
    numeric_providers = [p for p in providers if p["kind"] == "numeric"]
    fingerprint_providers = [p for p in providers if p["kind"] == "fingerprint"]
    embedding_providers = [p for p in providers if p["kind"] == "embedding"]
    
    st.subheader("🧪 Select Models")
    
    selected_providers = []
    provider_params = {}
    
    # Category tabs
    tab_numeric, tab_fp, tab_emb = st.tabs([
        f"📈 Numeric ({len(numeric_providers)})",
        f"🔢 Fingerprint ({len(fingerprint_providers)})",
        f"🤖 Embedding ({len(embedding_providers)})"
    ])
    
    def render_selection(provider_list, tab_key):
        if not provider_list:
            st.info("No providers in this category")
            return [], {}
        
        options = {p["display_name"]: p for p in provider_list}
        selected_names = st.multiselect(
            "Select models",
            options=list(options.keys()),
            key=f"select_{tab_key}"
        )
        
        selected = [options[name] for name in selected_names]
        params = {}
        
        if selected:
            st.divider()
            for p in selected:
                with st.expander(f"{p['display_name']} settings", expanded=True):
                    p_params = {}
                    for spec in p.get("params_schema", []):
                        if spec["type"] == "int":
                            p_params[spec["name"]] = st.number_input(
                                spec["name"],
                                value=spec.get("default", 0),
                                key=f"{tab_key}_{p['name']}_{spec['name']}"
                            )
                        elif spec["type"] == "select":
                            opts = spec.get("options", [])
                            p_params[spec["name"]] = st.selectbox(
                                spec["name"],
                                options=opts,
                                index=opts.index(spec.get("default")) if spec.get("default") in opts else 0,
                                key=f"{tab_key}_{p['name']}_{spec['name']}"
                            )
                        elif spec["type"] == "bool":
                            p_params[spec["name"]] = st.checkbox(
                                spec["name"],
                                value=spec.get("default", False),
                                key=f"{tab_key}_{p['name']}_{spec['name']}"
                            )
                    params[p["name"]] = p_params
        
        return selected, params
    
    with tab_numeric:
        sel, par = render_selection(numeric_providers, "numeric")
        selected_providers.extend(sel)
        provider_params.update(par)
    
    with tab_fp:
        sel, par = render_selection(fingerprint_providers, "fp")
        selected_providers.extend(sel)
        provider_params.update(par)
    
    with tab_emb:
        sel, par = render_selection(embedding_providers, "emb")
        selected_providers.extend(sel)
        provider_params.update(par)
    
    st.divider()
    
    if selected_providers:
        st.success(f"✅ {len(selected_providers)} model(s) selected")
    else:
        st.warning("Please select at least one model")
    
    # Run button
    if st.button("🚀 Generate Descriptors", type="primary", use_container_width=True, disabled=not selected_providers):
        # Get valid SMILES
        valid_smiles = [
            r["smiles_normalized"] or r["input_smiles_raw"]
            for r in result["records"]
            if r["parse_status"] == "OK"
        ]
        
        with st.spinner("Generating descriptors via API..."):
            api_result = api_request("POST", "/api/descriptors/generate", {
                "smiles_list": valid_smiles,
                "providers": [p["name"] for p in selected_providers],
                "params": provider_params,
                "use_cache": True
            })
            
            if api_result:
                st.session_state.results = api_result["results"]
                st.success(f"✅ Generated descriptors from {len(api_result['results'])} model(s)")
    
    if st.session_state.results:
        if st.button("➡️ View Results", type="primary"):
            st.session_state.page = "results"
            st.rerun()


def render_results_page():
    """Render results page"""
    st.header("3️⃣ Results & Export")
    
    if not st.session_state.results:
        st.warning("No results available. Please run descriptor generation first.")
        if st.button("← Go to Model Selection"):
            st.session_state.page = "models"
            st.rerun()
        return
    
    results = st.session_state.results
    
    # Summary
    st.subheader("📊 Summary")
    summary_data = []
    for name, data in results.items():
        summary_data.append({
            "Model": data["display_name"],
            "Type": data["kind"],
            "Success": data["success_count"],
            "Errors": data["error_count"],
            "Features": len(data["feature_columns"]),
            "Time (s)": round(data["execution_time_seconds"], 2)
        })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # Tabs for each model
    st.subheader("📋 Detailed Results")
    
    tabs = st.tabs([data["display_name"] for data in results.values()])
    
    for tab, (name, data) in zip(tabs, results.items()):
        with tab:
            col1, col2, col3 = st.columns(3)
            col1.metric("Success", data["success_count"])
            col2.metric("Errors", data["error_count"])
            col3.metric("Features", len(data["feature_columns"]))
            
            # Features
            if data["features"]:
                features_df = pd.DataFrame(data["features"])
                
                if len(features_df.columns) > 20:
                    st.warning(f"⚠️ {len(features_df.columns)} columns - showing first 20")
                    st.dataframe(features_df.iloc[:, :20], use_container_width=True)
                else:
                    st.dataframe(features_df, use_container_width=True)
                
                # Download
                st.subheader("📥 Download")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = features_df.to_csv(index=False)
                    st.download_button(
                        "📄 Download CSV",
                        data=csv,
                        file_name=f"{name}_features.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    import json
                    json_data = json.dumps(data["features"], indent=2)
                    st.download_button(
                        "📋 Download JSON",
                        data=json_data,
                        file_name=f"{name}_features.json",
                        mime="application/json",
                        use_container_width=True
                    )


def main():
    """Main application entry point"""
    render_sidebar()
    
    if st.session_state.page == "input":
        render_input_page()
    elif st.session_state.page == "models":
        render_model_page()
    elif st.session_state.page == "results":
        render_results_page()


if __name__ == "__main__":
    main()
