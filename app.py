"""
Polymer SMILES Descriptor Generator

A Streamlit application for generating molecular descriptors from polymer SMILES.
Supports RDKit 2D descriptors, Morgan fingerprints, and Transformer embeddings.
"""
import streamlit as st
import pandas as pd
import io

from config import APP_NAME, APP_VERSION
from core.parsing import parse_smiles_input, ParseStatus
from core.cache import get_cache
from providers.registry import ProviderRegistry, register_all_providers
from export_io.export import get_export_filename


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


def init_providers():
    """Initialize descriptor providers"""
    if not ProviderRegistry.get_names():
        register_all_providers()


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
            "results": "3️⃣ View Results",
            "atom_viz": "🔬 Atom Attribution"
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
            st.metric("Total", result.total_count)
            col1, col2 = st.columns(2)
            col1.metric("✅ Valid", result.success_count)
            col2.metric("❌ Error", result.error_count)
            if result.polymer_count > 0:
                st.metric("🔗 Polymer (*)", result.polymer_count)
        
        st.divider()
        
        # Cache stats
        cache = get_cache()
        stats = cache.get_stats()
        if stats["num_entries"] > 0:
            st.subheader("💾 Cache")
            st.caption(f"{stats['num_entries']} entries ({stats['total_size_mb']} MB)")
            if st.button("Clear Cache", use_container_width=True):
                cache.clear()
                st.toast("Cache cleared!")


def render_input_page():
    """Render SMILES input page"""
    st.header("1️⃣ Input Polymer SMILES")
    
    st.markdown("""
    Enter polymer SMILES or copolymer compositions.
    Polymer wildcards (`*`) are supported and will be handled according to settings.
    """)
    
    # Input method tabs
    input_mode = st.radio(
        "Input Mode",
        options=["Single Polymer SMILES", "Copolymer (Monomer + Composition)"],
        horizontal=True,
        key="input_mode"
    )
    
    smiles_input = ""
    copolymer_data = []
    
    if input_mode == "Single Polymer SMILES":
        # Standard SMILES input
        tab1, tab2 = st.tabs(["📝 Text Input", "📁 File Upload"])
        
        with tab1:
            smiles_input = st.text_area(
                "SMILES Input",
                height=200,
                placeholder="CC(C)CC(=O)O\n*CC(C)(C)C*\nmol_001,c1ccccc1\nmol_002,CCO",
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
    
    else:
        # Copolymer input mode
        st.subheader("🔗 Copolymer Composition Input")
        st.caption("Enter monomer SMILES and their molar fractions. Descriptors = weighted average.")
        
        num_copolymers = st.number_input("Number of copolymer entries", min_value=1, max_value=100, value=1, key="num_copolymers")
        
        for i in range(int(num_copolymers)):
            with st.expander(f"Copolymer #{i+1}", expanded=(i == 0)):
                copolymer_id = st.text_input("ID", value=f"copoly_{i+1:03d}", key=f"copoly_id_{i}")
                num_monomers = st.number_input("Number of monomers", min_value=2, max_value=10, value=2, key=f"num_monomers_{i}")
                
                monomers = []
                total_frac = 0.0
                
                cols = st.columns([3, 1])
                cols[0].markdown("**Monomer SMILES**")
                cols[1].markdown("**Molar Fraction**")
                
                for j in range(int(num_monomers)):
                    cols = st.columns([3, 1])
                    with cols[0]:
                        monomer_smiles = st.text_input(f"Monomer {j+1}", placeholder="*CC(C)(C)*", key=f"monomer_{i}_{j}", label_visibility="collapsed")
                    with cols[1]:
                        frac = st.number_input(f"Frac {j+1}", min_value=0.0, max_value=1.0, value=1.0/num_monomers, step=0.05, key=f"frac_{i}_{j}", label_visibility="collapsed")
                    
                    if monomer_smiles:
                        monomers.append({"smiles": monomer_smiles, "fraction": frac})
                        total_frac += frac
                
                if monomers and abs(total_frac - 1.0) > 0.01:
                    st.warning(f"⚠️ Fractions sum to {total_frac:.3f} (should be 1.0)")
                
                if monomers:
                    copolymer_data.append({"id": copolymer_id, "monomers": monomers})
        
        st.session_state.copolymer_entries = copolymer_data
    
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
            help="replace: Convert * to [*] for RDKit compatibility\nskip: Keep as-is\nerror: Raise error"
        )
    
    # Parse button
    if st.button("🔍 Parse & Validate", type="primary", use_container_width=True):
        if input_mode == "Single Polymer SMILES":
            if not smiles_input.strip():
                st.error("Please enter at least one SMILES string")
            else:
                with st.spinner("Parsing SMILES..."):
                    result = parse_smiles_input(
                        smiles_input,
                        canonicalize=canonicalize,
                        wildcard_mode=wildcard_mode
                    )
                    st.session_state.parse_result = result
                    st.session_state.input_type = "single"
                
                if result.total_count == 0:
                    st.error("No valid SMILES found in input")
        else:
            # Copolymer mode
            if not copolymer_data:
                st.error("Please enter at least one copolymer with monomers")
            else:
                all_monomer_smiles = []
                for copoly in copolymer_data:
                    for m in copoly["monomers"]:
                        all_monomer_smiles.append(m["smiles"])
                
                smiles_text = "\n".join(all_monomer_smiles)
                with st.spinner("Parsing monomer SMILES..."):
                    result = parse_smiles_input(
                        smiles_text,
                        canonicalize=canonicalize,
                        wildcard_mode=wildcard_mode
                    )
                    st.session_state.parse_result = result
                    st.session_state.input_type = "copolymer"
                    st.session_state.copolymer_data = copolymer_data
                
                st.success(f"✅ Parsed {len(all_monomer_smiles)} monomers from {len(copolymer_data)} copolymers")
    
    # Show results if we have them (persists across reruns)
    if st.session_state.parse_result:
        result = st.session_state.parse_result
        
        # Show summary
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Entries", result.total_count)
        col2.metric("Successfully Parsed", result.success_count)
        col3.metric("Errors", result.error_count)
        
        # Show results table
        if result.records:
            st.subheader("📋 Parse Results")
            
            df_data = []
            for r in result.records:
                df_data.append({
                    "ID": r.input_id,
                    "Input SMILES": r.input_smiles_raw[:50] + "..." if len(r.input_smiles_raw) > 50 else r.input_smiles_raw,
                    "Status": "✅" if r.parse_status == ParseStatus.OK else "❌",
                    "Polymer (*)": "Yes" if r.has_polymer_wildcard else "",
                    "Error": r.error_message[:30] if r.error_message else ""
                })
            
            st.dataframe(pd.DataFrame(df_data), use_container_width=True)
        
        if result.success_count > 0:
            st.success(f"✅ {result.success_count} SMILES ready for descriptor generation")
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
    st.info(f"📊 {result.success_count} SMILES entries ready for processing")
    
    # Initialize providers
    init_providers()
    providers = ProviderRegistry.list_all()
    
    if not providers:
        st.error("No descriptor providers available!")
        return
    
    # Group providers by kind
    numeric_providers = [p for p in providers if p.kind == "numeric"]
    fingerprint_providers = [p for p in providers if p.kind == "fingerprint"]
    embedding_providers = [p for p in providers if p.kind == "embedding"]
    
    # Model selection with categories
    st.subheader("🧪 Select Models")
    
    selected_providers = []
    provider_params = {}
    
    # Use tabs for categories
    tab_numeric, tab_fingerprint, tab_embedding = st.tabs([
        f"📈 Numeric ({len(numeric_providers)})",
        f"🔢 Fingerprint ({len(fingerprint_providers)})",
        f"🤖 Embedding ({len(embedding_providers)})"
    ])
    
    def render_provider_selection(providers_list, tab_key):
        """Render provider selection within a tab"""
        selected = []
        
        if not providers_list:
            st.info("No providers available in this category")
            return selected
        
        # Create option mapping
        options = {p.display_name: p for p in providers_list}
        
        # Multiselect for quick selection
        selected_names = st.multiselect(
            "Select models",
            options=list(options.keys()),
            default=[],
            key=f"multiselect_{tab_key}",
            help="Select one or more models to enable"
        )
        
        selected = [options[name] for name in selected_names]
        
        # Quick info cards in columns
        if providers_list:
            cols = st.columns(min(len(providers_list), 3))
            for i, provider in enumerate(providers_list):
                with cols[i % 3]:
                    is_selected = provider.display_name in selected_names
                    polymer_badge = "🔗" if provider.supports_polymer_smiles else ""
                    
                    with st.container(border=True):
                        st.markdown(f"**{provider.display_name}** {polymer_badge}")
                        st.caption(f"{provider.kind} • v{provider.version}")
                        if is_selected:
                            st.success("✓ Selected", icon="✅")
        
        return selected
    
    def render_params_ui(provider, tab_key):
        """Render parameter UI for a provider"""
        params = {}
        schema = provider.params_schema()
        
        if not schema:
            st.caption("No configurable parameters")
            return params
        
        for spec in schema:
            if spec.type == "int":
                params[spec.name] = st.number_input(
                    spec.name,
                    value=spec.default,
                    min_value=int(spec.min_value) if spec.min_value else None,
                    max_value=int(spec.max_value) if spec.max_value else None,
                    help=spec.description,
                    key=f"{tab_key}_{provider.name}_{spec.name}"
                )
            elif spec.type == "float":
                params[spec.name] = st.number_input(
                    spec.name,
                    value=float(spec.default),
                    min_value=spec.min_value,
                    max_value=spec.max_value,
                    help=spec.description,
                    key=f"{tab_key}_{provider.name}_{spec.name}"
                )
            elif spec.type == "bool":
                params[spec.name] = st.checkbox(
                    spec.name,
                    value=spec.default,
                    help=spec.description,
                    key=f"{tab_key}_{provider.name}_{spec.name}"
                )
            elif spec.type == "select":
                params[spec.name] = st.selectbox(
                    spec.name,
                    options=spec.options,
                    index=spec.options.index(spec.default) if spec.default in spec.options else 0,
                    help=spec.description,
                    key=f"{tab_key}_{provider.name}_{spec.name}"
                )
            elif spec.type == "str":
                params[spec.name] = st.text_input(
                    spec.name,
                    value=spec.default,
                    help=spec.description,
                    key=f"{tab_key}_{provider.name}_{spec.name}"
                )
        
        return params
    
    with tab_numeric:
        selected_numeric = render_provider_selection(numeric_providers, "numeric")
        selected_providers.extend(selected_numeric)
        
        if selected_numeric:
            st.divider()
            st.markdown("**⚙️ Parameters**")
            for provider in selected_numeric:
                with st.expander(f"{provider.display_name} settings", expanded=True):
                    provider_params[provider.name] = render_params_ui(provider, "numeric")
    
    with tab_fingerprint:
        selected_fp = render_provider_selection(fingerprint_providers, "fingerprint")
        selected_providers.extend(selected_fp)
        
        if selected_fp:
            st.divider()
            st.markdown("**⚙️ Parameters**")
            for provider in selected_fp:
                with st.expander(f"{provider.display_name} settings", expanded=True):
                    provider_params[provider.name] = render_params_ui(provider, "fingerprint")
    
    with tab_embedding:
        selected_emb = render_provider_selection(embedding_providers, "embedding")
        selected_providers.extend(selected_emb)
        
        if selected_emb:
            st.divider()
            st.markdown("**⚙️ Parameters**")
            for provider in selected_emb:
                with st.expander(f"{provider.display_name} settings", expanded=True):
                    provider_params[provider.name] = render_params_ui(provider, "embedding")
    
    # Summary of selection
    st.divider()
    if selected_providers:
        st.success(f"✅ {len(selected_providers)} model(s) selected: {', '.join([p.display_name for p in selected_providers])}")
    else:
        st.warning("Please select at least one model")
    
    # Execution settings
    st.subheader("⚙️ Execution Settings")
    col1, col2 = st.columns(2)
    with col1:
        use_cache = st.checkbox("Use Cache", value=True, help="Cache results for faster re-runs")
    
    # Run button
    st.divider()
    
    if st.button("🚀 Generate Descriptors", type="primary", use_container_width=True):
        valid_records = result.get_valid_records()
        
        if not valid_records:
            st.error("No valid SMILES records to process")
            return
        
        progress_bar = st.progress(0, text="Starting...")
        results = {}
        cache = get_cache() if use_cache else None
        
        for i, provider in enumerate(selected_providers):
            progress_bar.progress(
                (i) / len(selected_providers),
                text=f"Processing {provider.display_name}..."
            )
            
            params = provider_params.get(provider.name, {})
            
            # Check if provider supports the SMILES
            if not provider.supports_polymer_smiles:
                # Filter to RDKit-compatible records
                records_to_process = [r for r in valid_records if r.is_rdkit_compatible()]
            else:
                records_to_process = valid_records
            
            if not records_to_process:
                st.warning(f"⚠️ No compatible SMILES for {provider.display_name}")
                continue
            
            # Check cache
            smiles_list = [r.smiles_normalized or r.input_smiles_raw for r in records_to_process]
            cache_key = cache.make_key(smiles_list, provider.name, params) if cache else None
            
            if cache and cache.has(cache_key):
                result_data = cache.get(cache_key)
                st.toast(f"📦 Loaded {provider.display_name} from cache")
            else:
                # Run featurization
                result_data = provider.featurize(records_to_process, params)
                if cache:
                    cache.set(cache_key, result_data)
            
            results[provider.name] = {
                "provider": provider,
                "result": result_data,
                "params": params
            }
        
        progress_bar.progress(1.0, text="Complete!")
        st.session_state.results = results
        
        st.success(f"✅ Generated descriptors from {len(results)} model(s)")
    
    # Show View Results button if we have results (persists across reruns)
    if st.session_state.results:
        if st.button("➡️ View Results", type="primary", key="view_results_btn"):
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
        result = data["result"]
        summary_data.append({
            "Model": data["provider"].display_name,
            "Type": data["provider"].kind,
            "Success": result.success_count,
            "Errors": result.error_count,
            "Features": len(result.features_df.columns),
            "Time (s)": round(result.execution_time_seconds, 2)
        })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # Tabs for each model
    st.subheader("📋 Detailed Results")
    
    tabs = st.tabs([data["provider"].display_name for data in results.values()])
    
    for tab, (name, data) in zip(tabs, results.items()):
        with tab:
            result = data["result"]
            provider = data["provider"]
            
            # Info
            col1, col2, col3 = st.columns(3)
            col1.metric("Success", result.success_count)
            col2.metric("Errors", result.error_count)
            col3.metric("Features", len(result.features_df.columns))
            
            # Meta tab and Features tab
            meta_tab, feature_tab, preview_tab = st.tabs(["📝 Meta", "📊 Features", "👁️ Preview"])
            
            with meta_tab:
                if not result.meta_df.empty:
                    st.dataframe(result.meta_df, use_container_width=True)
                else:
                    st.info("No metadata available")
            
            with feature_tab:
                if not result.features_df.empty:
                    # For high-dimensional data, show preview
                    if len(result.features_df.columns) > 20:
                        st.warning(f"⚠️ {len(result.features_df.columns)} columns - showing first 20")
                        st.dataframe(result.features_df.iloc[:, :20], use_container_width=True)
                    else:
                        st.dataframe(result.features_df, use_container_width=True)
                else:
                    st.info("No features generated")
            
            with preview_tab:
                if not result.features_df.empty:
                    st.caption("Statistical summary of features")
                    st.dataframe(result.features_df.describe(), use_container_width=True)
                else:
                    st.info("No features to preview")
            
            # Download buttons
            st.divider()
            st.subheader("📥 Download")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV download
                csv_buffer = io.StringIO()
                combined_df = result.to_combined_df()
                combined_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📄 Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name=get_export_filename(name, "csv"),
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Parquet download
                parquet_buffer = io.BytesIO()
                combined_df.to_parquet(parquet_buffer, index=False)
                st.download_button(
                    "📦 Download Parquet",
                    data=parquet_buffer.getvalue(),
                    file_name=get_export_filename(name, "parquet"),
                    mime="application/octet-stream",
                    use_container_width=True
                )
            
            with col3:
                # JSON download
                import json
                json_data = {
                    "metadata": result.run_meta,
                    "data": combined_df.to_dict(orient="records")
                }
                st.download_button(
                    "📋 Download JSON",
                    data=json.dumps(json_data, indent=2),
                    file_name=get_export_filename(name, "json"),
                    mime="application/json",
                    use_container_width=True
                )
            
            # Run metadata
            with st.expander("🔧 Run Metadata"):
                st.json(result.run_meta)


def render_atom_viz_page():
    """Render atom attribution visualization page"""
    st.header("🔬 Atom Attribution Visualization")
    
    st.markdown("""
    Visualize which atoms contribute most to molecular fingerprints.
    Atoms are colored by their importance (red = high, blue = low).
    """)
    
    # Import visualization modules
    try:
        from core.atom_attribution import get_atom_contributions
        from core.mol_visualizer import generate_attribution_image, image_to_base64
        VIZ_AVAILABLE = True
    except ImportError as e:
        st.error(f"Visualization modules not available: {e}")
        VIZ_AVAILABLE = False
        return
    
    # Input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        smiles_input = st.text_input(
            "SMILES",
            value="c1ccccc1O",
            placeholder="Enter a SMILES string",
            help="Enter a valid SMILES to visualize atom contributions"
        )
    
    with col2:
        method = st.selectbox(
            "Method",
            options=["morgan", "atompair"],
            index=0,
            help="morgan: Morgan fingerprint bit contributions\natompair: Atom pair fingerprint"
        )
    
    # Method-specific options
    if method == "morgan":
        col1, col2 = st.columns(2)
        with col1:
            radius = st.slider("Radius", 1, 4, 2)
        with col2:
            n_bits = st.selectbox("Bits", [512, 1024, 2048, 4096], index=2)
    else:
        radius = 2
        n_bits = 2048
    
    # Colormap
    colormap = st.selectbox(
        "Color scheme",
        options=["coolwarm", "RdYlGn", "viridis", "plasma"],
        index=0
    )
    
    # Generate button
    if st.button("🎨 Generate Visualization", type="primary", use_container_width=True):
        if not smiles_input.strip():
            st.error("Please enter a SMILES string")
            return
        
        try:
            with st.spinner("Calculating atom contributions..."):
                # Get contributions
                if method == "morgan":
                    result = get_atom_contributions(
                        smiles_input,
                        method="morgan",
                        radius=radius,
                        n_bits=n_bits
                    )
                else:
                    result = get_atom_contributions(
                        smiles_input,
                        method="atompair",
                        n_bits=n_bits
                    )
                
                # Generate image
                image_bytes = generate_attribution_image(
                    smiles_input,
                    result["atom_contributions"],
                    width=500,
                    height=400,
                    colormap=colormap
                )
                
                # Display
                st.subheader("Atom Contributions")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(image_bytes, caption=f"SMILES: {smiles_input}")
                
                with col2:
                    st.markdown("**Contribution Scores**")
                    atoms_df = pd.DataFrame({
                        "Atom": list(range(result["n_atoms"])),
                        "Score": [round(s, 3) for s in result["atom_contributions"]]
                    })
                    st.dataframe(atoms_df, use_container_width=True, hide_index=True)
                
                # Colorbar legend
                st.markdown("""
                <div style="display: flex; align-items: center; margin-top: 10px;">
                    <span style="margin-right: 10px;">Low</span>
                    <div style="width: 200px; height: 20px; background: linear-gradient(to right, 
                        rgb(59, 76, 192), rgb(255, 255, 255), rgb(180, 4, 38));
                        border-radius: 3px;"></div>
                    <span style="margin-left: 10px;">High</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.success(f"✅ Visualized {result['n_atoms']} atoms")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")


def main():
    """Main application entry point"""
    init_providers()
    render_sidebar()
    
    if st.session_state.page == "input":
        render_input_page()
    elif st.session_state.page == "models":
        render_model_page()
    elif st.session_state.page == "results":
        render_results_page()
    elif st.session_state.page == "atom_viz":
        render_atom_viz_page()


if __name__ == "__main__":
    main()
