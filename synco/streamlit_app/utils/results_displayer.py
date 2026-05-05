import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any
import json
import plotly.graph_objects as go
import plotly.express as px


def display_artifacts_debug(artifacts: Optional[Dict[str, Any]]):
    """Display debug info about artifacts structure."""
    if not artifacts:
        return

    st.write("**Artifacts Structure Debug:**")
    for key, value in artifacts.items():
        if isinstance(value, pd.DataFrame):
            st.write(f"- {key}: DataFrame ({value.shape[0]} rows, {value.shape[1]} cols)")
        elif isinstance(value, dict):
            st.write(f"- {key}: dict with keys: {list(value.keys())[:5]}...")
        elif isinstance(value, list):
            st.write(f"- {key}: list ({len(value)} items)")
        elif isinstance(value, str):
            st.write(f"- {key}: str (length {len(value)})")
        else:
            st.write(f"- {key}: {type(value).__name__}")


def display_results_summary(artifacts: Optional[Dict[str, Any]]):
    """Display overview of pipeline results."""
    if not artifacts:
        st.warning("No results available yet.")
        return

    st.subheader("📊 Results Summary")

    # Try to get global summary from skipped_info first (where actual results are stored)
    summary = None
    if "skipped_info" in artifacts and isinstance(artifacts["skipped_info"], dict):
        summary = artifacts["skipped_info"].get("global_summary")

    # Fall back to comparison_summary
    if summary is None and "comparison_summary" in artifacts:
        summary = artifacts["comparison_summary"]

    if summary:
        # Extract metrics
        tp = summary.get("Global True Positives", summary.get("TP", 0))
        tn = summary.get("Global True Negatives", summary.get("TN", 0))
        fp = summary.get("Global False Positives", summary.get("FP", 0))
        fn = summary.get("Global False Negatives", summary.get("FN", 0))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("TP", tp)
        with col2:
            st.metric("TN", tn)
        with col3:
            st.metric("FP", fp)
        with col4:
            st.metric("FN", fn)

        st.divider()

        # Calculate and display metrics
        total = tp + tn + fp + fn
        if total > 0:
            accuracy = summary.get("Global Accuracy %", (tp + tn) / total * 100) / 100 if isinstance(summary.get("Global Accuracy %"), (int, float)) else (tp + tn) / total
            precision = summary.get("Global Precision %", tp / (tp + fp) * 100 if (tp + fp) > 0 else 0) / 100 if isinstance(summary.get("Global Precision %"), (int, float)) else (tp / (tp + fp) if (tp + fp) > 0 else 0)
            recall = summary.get("Global Recall %", tp / (tp + fn) * 100 if (tp + fn) > 0 else 0) / 100 if isinstance(summary.get("Global Recall %"), (int, float)) else (tp / (tp + fn) if (tp + fn) > 0 else 0)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy:.3f}")
            with col2:
                st.metric("Precision", f"{precision:.3f}")
            with col3:
                st.metric("Recall", f"{recall:.3f}")
            with col4:
                st.metric("F1 Score", f"{f1:.3f}")
    else:
        st.info("Summary metrics not available yet")


def display_pair_details(artifacts: Optional[Dict[str, Any]]):
    """Display detailed pair comparison results."""
    if not artifacts or "pair_details" not in artifacts:
        return

    st.subheader("📋 Pair Comparison Details")

    pair_df = artifacts["pair_details"]
    if isinstance(pair_df, pd.DataFrame):
        st.dataframe(pair_df, width='stretch')
        st.caption(f"Total pairs analyzed: {len(pair_df)}")

        # Note about None values
        with st.expander("ℹ️ About pd_combination None values", expanded=False):
            st.info(
                "**None values in pd_combination** are expected and normal. "
                "This column contains the drug profile combination data from the prediction data. "
                "If no matching profile was found for a particular drug combination and cell line pair, "
                "the value will be None. This doesn't indicate an error in the analysis."
            )
    else:
        st.info("Pair details format not recognized")


def display_skipped_info(artifacts: Optional[Dict[str, Any]]):
    """Display information about skipped entries."""
    if not artifacts or "skipped_info" not in artifacts:
        return

    skipped = artifacts["skipped_info"]
    if not skipped:
        st.success("✅ No issues during comparison")
        return

    st.subheader("ℹ️ Comparison Summary")

    # If skipped_info is a dict with global_summary, display those metrics
    if isinstance(skipped, dict):
        if "global_summary" in skipped:
            summary = skipped["global_summary"]
            st.write("**Global Comparison Results:**")

            # Create two columns for metrics
            col1, col2 = st.columns(2)
            with col1:
                for key in ["Global matches", "Global matches %", "Global mismatches", "Global mismatches %"]:
                    if key in summary:
                        st.write(f"- **{key}**: {summary[key]}")
            with col2:
                for key in ["Global True Positives", "Global True Negatives", "Global False Positives", "Global False Negatives"]:
                    if key in summary:
                        st.write(f"- **{key}**: {summary[key]}")

        # Show detailed skipped information if any
        if "skipped_from_experimental" in skipped or "skipped_from_predicted" in skipped:
            with st.expander("View skipped pair details", expanded=False):
                if "skipped_from_experimental" in skipped:
                    st.write(f"**Skipped from experimental**: {len(skipped['skipped_from_experimental'])} pairs")
                if "skipped_from_predicted" in skipped:
                    st.write(f"**Skipped from predicted**: {len(skipped['skipped_from_predicted'])} pairs")
    elif isinstance(skipped, list) and len(skipped) > 0:
        with st.expander("View skipped entries", expanded=False):
            for idx, skip_info in enumerate(skipped):
                st.write(f"**Entry {idx + 1}:**")
                if isinstance(skip_info, dict):
                    for key, value in skip_info.items():
                        st.write(f"- {key}: {value}")
                elif isinstance(skip_info, str):
                    st.write(f"- {skip_info}")
                else:
                    st.write(f"- {str(skip_info)}")
                st.divider()
    else:
        st.success("✅ No pairs were skipped during comparison")


def display_synergy_predictions(artifacts: Optional[Dict[str, Any]]):
    """Display synergy predictions."""
    if not artifacts or "synergy_predictions" not in artifacts:
        return

    st.subheader("🔮 Synergy Predictions")

    pred_results = artifacts["synergy_predictions"]
    if isinstance(pred_results, dict):
        st.json(pred_results if len(str(pred_results)) < 1000 else {"info": "Large prediction dataset"})
    elif isinstance(pred_results, pd.DataFrame):
        st.dataframe(pred_results, width='stretch')
    else:
        st.info("Prediction format not recognized")


def display_convergence_results(artifacts: Optional[Dict[str, Any]]):
    """Display convergence analysis results - should show harmonized synergy tables."""
    if not artifacts:
        return

    st.subheader("🔄 Convergence Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Experimental Convergence**")
        if "experimental_convergence" in artifacts:
            exp_conv = artifacts["experimental_convergence"]
            if isinstance(exp_conv, pd.DataFrame):
                st.write(f"DataFrame: {exp_conv.shape[0]} rows × {exp_conv.shape[1]} columns")
                st.dataframe(exp_conv, width='stretch', use_container_width=True)
            elif isinstance(exp_conv, dict):
                st.write(f"Dictionary with {len(exp_conv)} items")
                with st.expander("View structure", expanded=True):
                    for key in list(exp_conv.keys())[:10]:
                        st.write(f"- {key}")
            elif isinstance(exp_conv, str):
                st.caption("Convergence data (text representation)")
                with st.expander("View data (first 500 chars)", expanded=False):
                    st.code(exp_conv[:500])
            else:
                st.info(f"Data type: {type(exp_conv).__name__}")
        else:
            st.info("No experimental convergence data")

    with col2:
        st.markdown("**Predictions Convergence**")
        if "predictions_convergence" in artifacts:
            pred_conv = artifacts["predictions_convergence"]
            if isinstance(pred_conv, pd.DataFrame):
                st.write(f"DataFrame: {pred_conv.shape[0]} rows × {pred_conv.shape[1]} columns")
                st.dataframe(pred_conv, width='stretch', use_container_width=True)
            elif isinstance(pred_conv, dict):
                st.write(f"Dictionary with {len(pred_conv)} items")
                with st.expander("View structure", expanded=True):
                    for key in list(pred_conv.keys())[:10]:
                        st.write(f"- {key}")
            elif isinstance(pred_conv, str):
                st.caption("Convergence data (text representation)")
                with st.expander("View data (first 500 chars)", expanded=False):
                    st.code(pred_conv[:500])
            else:
                st.info(f"Data type: {type(pred_conv).__name__}")
        else:
            st.info("No predictions convergence data")


def display_roc_results(artifacts: Optional[Dict[str, Any]]):
    """Display ROC analysis results with plots."""
    if not artifacts or "roc_results" not in artifacts:
        return

    st.subheader("📈 ROC Analysis Results")

    roc_results = artifacts["roc_results"]

    # Handle tuple format: (roc_tuples, pr_tuples, auc_list, pr_auc_list, df)
    if isinstance(roc_results, tuple) and len(roc_results) >= 5:
        roc_tuples = roc_results[0]  # ROC curve tuples
        pr_tuples = roc_results[1]   # PR curve tuples
        auc_list = roc_results[2]
        pr_auc_list = roc_results[3]
        roc_df = roc_results[-1]     # DataFrame

        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            mean_auc = sum(auc_list) / len(auc_list) if auc_list else 0
            st.metric("Mean AUC", f"{mean_auc:.3f}")
        with col2:
            mean_pr_auc = sum(pr_auc_list) / len(pr_auc_list) if pr_auc_list else 0
            st.metric("Mean PR-AUC", f"{mean_pr_auc:.3f}")
        with col3:
            st.metric("Cell Lines", len(roc_df) if isinstance(roc_df, pd.DataFrame) else len(auc_list))

        st.divider()

        # Create ROC curve plot if we have the tuples
        if roc_tuples and len(roc_tuples) > 0:
            st.write("**ROC Curves by Cell Line:**")

            fig_roc = go.Figure()

            # Add ROC curves for each cell line
            for idx, (auc_val, scatter_obj) in enumerate(roc_tuples):
                try:
                    if hasattr(scatter_obj, 'x') and hasattr(scatter_obj, 'y'):
                        # Extract cell line name from scatter name (e.g., "C2BBE1 (AUC=0.708)")
                        cell_line_name = scatter_obj.name.split(' (')[0] if ' (' in scatter_obj.name else f"Cell Line {idx}"

                        fig_roc.add_trace(go.Scatter(
                            x=scatter_obj.x,
                            y=scatter_obj.y,
                            mode='lines',
                            name=f"{cell_line_name} (AUC={auc_val:.3f})",
                            line=dict(width=2),
                            hovertemplate=(
                                "<b>%{fullData.name}</b><br>"
                                "FPR=%{x:.3f}, TPR=%{y:.3f}<br>"
                                "<extra></extra>"
                            )
                        ))
                except:
                    pass

            # Add diagonal reference line
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(dash='dash', color='gray', width=1)
            ))

            # Add average AUC annotation
            avg_auc = sum(auc_list) / len(auc_list) if auc_list else 0
            med_auc = sorted(auc_list)[len(auc_list)//2] if auc_list else 0

            fig_roc.add_annotation(
                x=0.5, y=1.08,
                text=f"Average ROC-AUC: {avg_auc:.4f} | Median ROC-AUC: {med_auc:.4f}",
                showarrow=False,
                font=dict(size=12),
                xref="paper", yref="paper"
            )

            fig_roc.update_layout(
                title="ROC Curves",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                hovermode='closest',
                height=600,
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )

            st.plotly_chart(fig_roc, width='stretch')

        # Create PR curve plot if we have the tuples
        if pr_tuples and len(pr_tuples) > 0:
            st.write("**Precision-Recall Curves by Cell Line:**")

            fig_pr = go.Figure()

            # Add PR curves for each cell line
            for idx, (pr_auc_val, scatter_obj) in enumerate(pr_tuples):
                try:
                    if hasattr(scatter_obj, 'x') and hasattr(scatter_obj, 'y'):
                        # Extract cell line name from scatter name
                        cell_line_name = scatter_obj.name.split(' (')[0] if ' (' in scatter_obj.name else f"Cell Line {idx}"

                        fig_pr.add_trace(go.Scatter(
                            x=scatter_obj.x,
                            y=scatter_obj.y,
                            mode='lines',
                            name=f"{cell_line_name} (PR-AUC={pr_auc_val:.3f})",
                            line=dict(width=2),
                            hovertemplate=(
                                "<b>%{fullData.name}</b><br>"
                                "Recall=%{x:.3f}, Precision=%{y:.3f}<br>"
                                "<extra></extra>"
                            )
                        ))
                except:
                    pass

            # Add average PR AUC annotation
            avg_pr_auc = sum(pr_auc_list) / len(pr_auc_list) if pr_auc_list else 0
            med_pr_auc = sorted(pr_auc_list)[len(pr_auc_list)//2] if pr_auc_list else 0

            fig_pr.add_annotation(
                x=0.5, y=1.08,
                text=f"Average PR-AUC: {avg_pr_auc:.4f} | Median PR-AUC: {med_pr_auc:.4f}",
                showarrow=False,
                font=dict(size=12),
                xref="paper", yref="paper"
            )

            fig_pr.update_layout(
                title="Precision-Recall Curves",
                xaxis_title="Recall",
                yaxis_title="Precision",
                hovermode='closest',
                height=600,
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )

            st.plotly_chart(fig_pr, width='stretch')

        st.divider()
        st.write("**ROC Results by Cell Line:**")
        if isinstance(roc_df, pd.DataFrame):
            st.dataframe(roc_df, width='stretch')

    elif isinstance(roc_results, dict):
        # Display ROC metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            auc = roc_results.get("auc", "N/A")
            st.metric("AUC", f"{auc:.3f}" if isinstance(auc, (int, float)) else auc)

        with col2:
            auc_pr = roc_results.get("auc_pr", "N/A")
            st.metric("AUC-PR", f"{auc_pr:.3f}" if isinstance(auc_pr, (int, float)) else auc_pr)

        with col3:
            threshold = roc_results.get("threshold", "N/A")
            st.metric("Threshold", f"{threshold:.4f}" if isinstance(threshold, (int, float)) else threshold)

        # Show additional ROC details
        with st.expander("ROC Details", expanded=False):
            for key, value in roc_results.items():
                if key not in ["auc", "auc_pr", "threshold", "fpr", "tpr", "precision", "recall"]:
                    if isinstance(value, (list, dict)):
                        st.write(f"**{key}**: {type(value).__name__} ({len(value)} items)")
                    else:
                        st.write(f"**{key}**: {value}")

    elif isinstance(roc_results, pd.DataFrame):
        st.dataframe(roc_results, width='stretch')

    elif isinstance(roc_results, str):
        st.write(roc_results)


def display_drug_profiles(artifacts: Optional[Dict[str, Any]]):
    """Display drug profile summary with selectable profiles."""
    if not artifacts or "drug_profiles" not in artifacts:
        return

    st.subheader("💊 Drug Profiles")

    profiles = artifacts["drug_profiles"]
    if isinstance(profiles, dict):
        st.write(f"**Total profiles loaded:** {len(profiles)}")

        # Display profile types
        profile_types = {}
        for drug_name, profile in profiles.items():
            ptype = type(profile).__name__
            profile_types[ptype] = profile_types.get(ptype, 0) + 1

        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("**Profile Types:**")
            for ptype, count in profile_types.items():
                st.write(f"- {ptype}: {count}")

        with col2:
            st.write("**Select profiles:**")
            selected_profiles = st.multiselect(
                "Profiles to view:",
                options=list(profiles.keys()),
                key="drug_profiles_select_key"
            )

        if selected_profiles:
            st.divider()
            for profile_name in selected_profiles:
                profile_data = profiles[profile_name]
                st.write(f"**{profile_name}** ({type(profile_data).__name__})")

                if isinstance(profile_data, pd.DataFrame):
                    st.dataframe(profile_data, width='stretch')
                elif isinstance(profile_data, dict):
                    with st.expander("View dict", expanded=False):
                        st.json(profile_data if len(str(profile_data)) < 1000 else {"info": "Large dict"})
                elif isinstance(profile_data, list):
                    st.write(f"List with {len(profile_data)} items")
                else:
                    st.code(str(profile_data)[:500])

    elif isinstance(profiles, pd.DataFrame):
        st.write(f"**Drug Profiles DataFrame:** {len(profiles)} rows")
        st.dataframe(profiles, width='stretch')

    elif isinstance(profiles, list):
        st.write(f"**Drug Profiles List:** {len(profiles)} items")
        with st.expander("View details", expanded=False):
            for i, profile in enumerate(list(profiles)[:5]):
                st.write(f"- [{i}]: {type(profile).__name__}")


def display_synergy_data(artifacts: Optional[Dict[str, Any]]):
    """Display synergy data summary with selectable cell lines."""
    if not artifacts or "synergy_data_dict" not in artifacts:
        return

    st.subheader("📊 Synergy Data")

    synergy_dict = artifacts["synergy_data_dict"]
    if isinstance(synergy_dict, dict):
        st.write(f"**Cell lines processed:** {len(synergy_dict)}")

        # Let user select which cell lines to view
        selected_cells = st.multiselect(
            "Select cell lines to view:",
            options=list(synergy_dict.keys()),
            key="synergy_data_select_key"
        )

        if selected_cells:
            st.divider()
            for cell_line in selected_cells:
                data = synergy_dict[cell_line]
                st.write(f"**{cell_line}** ({type(data).__name__})")

                if isinstance(data, pd.DataFrame):
                    st.write(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")
                    st.dataframe(data, width='stretch')

                elif isinstance(data, tuple):
                    st.write(f"Tuple with {len(data)} items")
                    col_select, col_info = st.columns([1, 1])

                    # Show tuple structure
                    with col_info:
                        st.write("**Tuple structure:**")
                        for i, item in enumerate(data):
                            if isinstance(item, pd.DataFrame):
                                st.write(f"- [{i}]: DataFrame ({item.shape[0]} rows)")
                            elif isinstance(item, dict):
                                st.write(f"- [{i}]: dict ({len(item)} keys)")
                            else:
                                st.write(f"- [{i}]: {type(item).__name__}")

                    # Let user select which tuple item to view
                    with col_select:
                        item_options = [f"Item {i}: {type(data[i]).__name__}" for i in range(len(data))]
                        selected_item_idx = st.selectbox(
                            f"View tuple item:",
                            range(len(data)),
                            format_func=lambda i: item_options[i],
                            key=f"synergy_tuple_item_{cell_line}"
                        )

                    # Display selected tuple item
                    item_data = data[selected_item_idx]
                    st.write(f"**Tuple Item {selected_item_idx}:**")
                    if isinstance(item_data, pd.DataFrame):
                        st.write(f"DataFrame: {item_data.shape[0]} rows × {item_data.shape[1]} columns")
                        st.dataframe(item_data, width='stretch')
                    elif isinstance(item_data, dict):
                        st.write(f"Dictionary with {len(item_data)} keys")
                        with st.expander("View keys", expanded=False):
                            for key in list(item_data.keys())[:30]:
                                st.write(f"- {key}")
                    else:
                        st.code(str(item_data)[:500])

                elif isinstance(data, dict):
                    st.write(f"Dictionary with {len(data)} keys")
                    with st.expander("View keys", expanded=False):
                        for key in list(data.keys())[:30]:
                            st.write(f"- {key}")

                else:
                    st.write(str(data)[:500])

    elif isinstance(synergy_dict, pd.DataFrame):
        st.write(f"**Synergy Data DataFrame:** {len(synergy_dict)} rows")
        st.dataframe(synergy_dict, width='stretch')

    elif isinstance(synergy_dict, list):
        st.write(f"**Synergy Data List:** {len(synergy_dict)} items")
        with st.expander("View details", expanded=False):
            for i, item in enumerate(list(synergy_dict)[:5]):
                st.write(f"- [{i}]: {type(item).__name__}")


def display_execution_summary(artifacts: Optional[Dict[str, Any]]):
    """Display execution summary."""
    st.subheader("⏱️ Pipeline Execution")

    st.write("""
    ✅ **Pipeline completed successfully!**

    The analysis included:
    - Synergy data loading and harmonization
    - Drug profile extraction
    - Synergy prediction extraction
    - Convergence analysis
    - Synergy comparison and classification
    - ROC/PR metrics calculation
    """)


def display_artifacts_overview(artifacts: Optional[Dict[str, Any]]):
    """Display what artifacts are available."""
    if not artifacts:
        return

    st.subheader("📦 Available Data")

    # Create tabs for each artifact type
    available_data = list(artifacts.keys())

    if not available_data:
        st.info("No artifacts available")
        return

    tabs = st.tabs(available_data)

    for idx, (tab, key) in enumerate(zip(tabs, available_data)):
        with tab:
            data = artifacts[key]

            if isinstance(data, pd.DataFrame):
                st.write(f"📊 **DataFrame**: {data.shape[0]} rows × {data.shape[1]} columns")
                st.dataframe(data, width='stretch')

            elif isinstance(data, dict):
                st.write(f"📋 **Dictionary**: {len(data)} keys")
                with st.expander("View keys", expanded=True):
                    cols = st.columns(3)
                    for i, key_name in enumerate(list(data.keys())[:30]):
                        col_idx = i % 3
                        with cols[col_idx]:
                            st.write(f"- {key_name}")

            elif isinstance(data, tuple):
                st.write(f"📝 **Tuple**: {len(data)} items")
                with st.expander("View structure", expanded=True):
                    for i, item in enumerate(data):
                        if isinstance(item, pd.DataFrame):
                            st.write(f"**Item {i}**: DataFrame ({item.shape[0]} rows, {item.shape[1]} cols)")
                            with st.expander(f"View Item {i}", expanded=False):
                                st.dataframe(item, width='stretch')
                        elif isinstance(item, (list, tuple)):
                            st.write(f"**Item {i}**: {type(item).__name__} ({len(item)} items)")
                            if len(item) > 0 and len(item) < 20:
                                with st.expander(f"View Item {i}", expanded=False):
                                    for j, subitem in enumerate(item[:5]):
                                        st.write(f"  - [{j}]: {type(subitem).__name__}")
                        elif isinstance(item, dict):
                            st.write(f"**Item {i}**: dict ({len(item)} keys)")
                        else:
                            st.write(f"**Item {i}**: {type(item).__name__}")

            elif isinstance(data, list):
                st.write(f"📝 **List**: {len(data)} items")
                if len(data) > 0:
                    with st.expander("View first 10 items", expanded=False):
                        for i, item in enumerate(data[:10]):
                            if isinstance(item, (dict, pd.DataFrame)):
                                st.write(f"Item {i}: {type(item).__name__}")
                            else:
                                st.write(f"Item {i}: {str(item)[:100]}")

            elif isinstance(data, str):
                st.write(f"📄 **String**: {len(data)} characters")
                with st.expander("View content", expanded=False):
                    st.text(data[:1000] + ("..." if len(data) > 1000 else ""))

            else:
                st.write(f"📄 **{type(data).__name__}**")
                with st.expander("View value", expanded=False):
                    st.write(str(data)[:500])
