import numpy as np
import pandas as pandas
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#//////////////////////////////////////////////////////////////////////////////

def plot_curves(
        traces, 
        auc_score_list, 
        model,
        specific_date, 
        tissue,
        metric='ROC', 
        width=800, 
        height=800,
        output=None
        ):
    
    fig = go.Figure()

    sorted_traces = sorted(traces, key=lambda x: x[0], reverse=True)
    num_traces = len(sorted_traces)

    # Add sorted traces to the figure
    for _, trace in sorted_traces:
        fig.add_trace(trace)

    # Add average auc_score annotation
    avg_auc_score = sum(auc_score_list) / len(auc_score_list)
    fig.add_annotation(
        x=0.5, y=1.05,
        text=f'Average {metric}-AUC Score: {avg_auc_score:.4f}',
        showarrow=False,
        font=dict(size=14)
    )

    # Add median auc_score annotation
    median_auc_score = np.median(auc_score_list)
    fig.add_annotation(
        x=0.5, y=1.02,
        text=f'Median {metric}-AUC Score: {median_auc_score:.4f}',
        showarrow=False,
        font=dict(size=14)
    )

    # Add model type and run date annotation
    if specific_date is None:
        fig.add_annotation(
            x=0.5, y=1.08,
            text=f'Model: {model} | Last run date',
            showarrow=False,
            font=dict(size=14)
        )
    else:
        fig.add_annotation(
            x=0.5, y=1.08,
            text=f'Model: {model} | Run Date: {specific_date}',
            showarrow=False,
            font=dict(size=14)
        )

    if metric == 'ROC':
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig.update_layout(
            title=f'{metric} Curve for {tissue} cell lines (n={num_traces})',
            xaxis_title='False Positive Rate (FPR)',
            yaxis_title='True Positive Rate (TPR)',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=width, height=height
        )
    elif metric == 'PR':
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0
        )
        fig.update_layout(
            title=f'{metric} Curve for {tissue} cell lines (n={num_traces})',
            xaxis_title='Recall',
            yaxis_title='Precision',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=width, height=height
        )

    if output:
        fig.write_html(output / f"{metric}_curve_{tissue}.html")
        fig.write_image(output / f"{metric}_curve_{tissue}.svg", scale=2)
        
    fig.show()


#//////////////////////////////////////////////////////////////////////////////

def plot_violin_and_table(
        selected_tissues, 
        violin_data, 
        auc_summary_table,
        model,
        specific_date,
        metric='ROC',
        metric_score='roc_auc_score',
        threshold=0.5,
        width=1200,
        height=700,
        tissue_colors = {'Breast': 'pink', 'Colon': 'rgb(136, 204, 238)', 'Pancreas': '#FF9E7A'},
        line_colors = {'Breast': '#FF6692', 'Colon': '#636EFA', 'Pancreas': '#EF553B'}
        ):

    # Create the figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.05,
        row_heights=[0.8, 0.2],
        specs=[[{"type": "violin"}], [{"type": "table"}]]
    )

    # Add violin plots for each tissue
    for tissue in selected_tissues:
        tissue_data = violin_data[violin_data['tissue'] == tissue]
        color = tissue_colors.get(tissue, 'blue') # Default to 'blue' if tissue not found in dictionary
        linecolor = line_colors.get(tissue, 'blue') # Default to 'blue' if tissue not found in dictionary
        fig.add_trace(
            go.Violin(
                y=tissue_data[metric_score],
                x=tissue_data['tissue'],
                meanline_visible=True,
                points='all',
                name=f'{tissue}',
                hovertext='Cell line: ' + tissue_data['cell_line_name'],
                hovertemplate='AUC: %{y:.2f}<extra></extra>, %{hovertext}, (%{x})',
                line_color=linecolor,
                fillcolor=color,
                opacity=0.8
            ),
            row=1, col=1
        )

    # Add threshold line
    fig.add_shape(
        name='threshold_line',
        label=dict(text=f'{metric}_THR = {threshold}', font_size=12, textposition='end'),
        type="line",
        x0=-0.5, x1=len(selected_tissues) - 0.5,
        y0=threshold, y1=threshold,
        line=dict(color="rgb(82, 106, 131)", width=2, dash="6px"),
        row=1, col=1
    )

    # Add table with summary statistics
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Tissue', 'Cell lines', 'Avg', 'Median', 'Max', 'Min', 'Std. Dev.', 'AUC>THR', 'P-value'],
                font=dict(size=12),
                align="center"
            ),
            cells=dict(
                values=[auc_summary_table.index] + [auc_summary_table[col].tolist() for col in auc_summary_table.columns],
                font=dict(size=12),
                align="center")
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=height,
        width=width,
        showlegend=True,
        title_text=f"Violin plot of {metric} AUC Scores by Tissue (model: {model}, run date: {specific_date})",
    )
    fig.show()


#//////////////////////////////////////////////////////////////////////////////

def plot_dots(
        auc_values,
        tissue_dict,
        tissue,
        show_scores=False,
        colorscale='Sunset',
        metric='ROC',
        threshold=0.5,
        height=800,
        width=500
):
    
    #Sort AUC values and cell line names for dots plot
    sorted_indices = sorted(range(len(auc_values)), key=lambda k: auc_values[k], reverse=False)
    sorted_auc_values = [auc_values[i] for i in sorted_indices]
    sorted_cell_lines = [tissue_dict[tissue][i] for i in sorted_indices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sorted_auc_values,
        y=sorted_cell_lines,
        mode='markers',
        name=f'{metric}_AUC',
        marker=dict(
            size=16,
            color=sorted_auc_values,  # set color equal to a variable
            colorscale= colorscale
            # showscale=True
        )
    ))

    # Add AUC threshold line in x=0.5
    fig.add_shape(
        type='line',
        x0=threshold,
        y0=0,
        x1=threshold,
        y1=len(tissue_dict[tissue]),
        line=dict(
            color='rgb(98, 83, 119)',
            width=3,
            dash='6px'
        )
    )

    # Add a dummy scatter for the threshold legend
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        name=f'AUC = {threshold}',
        line=dict(
            color='rgb(98, 83, 119)',  # Same color as the threshold line
            width=3,
            dash='dash'
        )
    ))

    if show_scores:
        # Annotations
        annotations = [
            dict(
                x=1.2,
                y=i,
                xref='paper',
                yref='y',
                text=f'AUC={auc:.3f}',
                showarrow=False,
                xanchor='right',
            ) for i, auc in enumerate(sorted_auc_values)
        ]

        fig.update_layout(
            annotations=annotations
        )

    fig.update_layout(
        title=f'{metric} AUC Scores for {tissue} cell line',
        xaxis_title='AUC Value',
        yaxis_title='Cell line',
        # showlegend=True,
        height=height,
        width=width,
        margin=dict(r=150)
    )

    fig.show()

def plot_multi_dots(
    auc_values_dict,
    metric='ROC',
    colorscale_by_auc=False,
    show_scores=False,
    threshold=0.5,
    height=800,
    width=500
):
    
    fig = go.Figure()

    # Group the dataframe by tissue
    grouped = auc_values_dict.groupby('tissue')

    for tissue, group in grouped:
        # Sort the group by AUC values
        group = group.sort_values(by='roc_auc_score')
        sorted_auc_values = group['roc_auc_score'].tolist()
        sorted_cell_lines = group['cell_line_name'].tolist()
        
        if colorscale_by_auc:
            fig.add_trace(go.Scatter(
                x=sorted_auc_values,
                y=sorted_cell_lines,
                legendgroup='AUC Scores',
                legendgrouptitle=dict(text='AUC Scores by tissue'),
                mode='markers',
                name=f'{tissue}',
                marker=dict(
                    size=12,
                    color=sorted_auc_values,  # set color equal to a variable
                    colorscale= 'Sunset'
                ),
            ))

        else:
            fig.add_trace(go.Scatter(
                x=sorted_auc_values,
                y=sorted_cell_lines,
                legendgroup='AUC Scores',
                legendgrouptitle=dict(text='AUC Scores by tissue'),
                name=f'{tissue}',
                mode='markers',
                marker=dict(
                    size=12
                ),
            ))

        if show_scores:
            # Annotations
            annotations = [
                dict(
                    x=1.2,  # Offset to the right
                    y=cell_line,
                    text=f'{auc:.3f}',
                    showarrow=False,
                    xanchor='left'
                ) for auc, cell_line in zip(sorted_auc_values, sorted_cell_lines)
            ]
            fig.update_layout(
                annotations=list(fig.layout.annotations) + annotations
            )

    # Add AUC threshold line in x=0.5
    fig.add_shape(
        type='line',
        x0=threshold,
        y0=0,
        x1=threshold,
        y1=len(auc_values_dict['cell_line_name'].unique()),
        line=dict(
            color='rgb(98, 83, 119)',  # Dark purple
            width=3,
            dash='dash'
        )
    )

    # Add a dummy scatter for the threshold legend
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        legendgroup='threshold',
        legendgrouptitle=dict(text='Threshold'),
        name=f'AUC = {threshold}',
        line=dict(
            color='rgb(98, 83, 119)',  # Same color as the threshold line
            width=3,
            dash='dash'
        )
    ))

    fig.update_layout(
        title=f'{metric} AUC Scores for Multiple Tissues',
        xaxis_title='AUC Value',
        yaxis_title='Cell line',
        showlegend=True,
        height=height,
        width=width,
        margin=dict(r=150)  # Increased right margin
    )

    fig.show()