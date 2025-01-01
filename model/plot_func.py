from pygwalker.api.gradio import PYGWALKER_ROUTE, get_html_on_gradio
import plotly.graph_objects as go
from model.setup_func import score_topic_for_text


def show_pygwalker(df):
    pyg_html = get_html_on_gradio(df, spec=r"utils/gw_config.json", spec_io_mode="r")
    return pyg_html

def plot_umap(plot_df, patient_text, model_path):
    fig = go.Figure()

    # 添加 UMAP 点的散点图
    fig.add_trace(
        go.Scattergl(
            x=plot_df["UMAP_1"],
            y=plot_df["UMAP_2"],
            mode="markers",
            marker=dict(
                size=1,  # 点的大小
                opacity=0.6,  # 点的透明度
                color=plot_df["Cluster"],  # 设置颜色映射
                colorscale="Agsunset",  # 设置颜色映射的色阶
                line=dict(width=0),  # 去掉点的边框
                symbol="circle",  # 使用圆形标记
            ),
            name="UMAP Points",  # 设置图例名称
            showlegend=True,  # 确保图例显示
            legendgroup="Cluster",  # 使用相同的 legendgroup 确保它们在图例中合并
        )
    )

    # 获取患者点坐标并高亮
    # patient_id = int(patient_id)
    # patient_row = plot_df.iloc[patient_id]
    predicted_topic, score, nmf_2d = score_topic_for_text(model_path, patient_text)
    umap_1, umap_2 = nmf_2d[0, 0] , nmf_2d[0, 1]

    # 添加高亮患者点的散点图
    fig.add_trace(
        go.Scattergl(
            x=[umap_1],
            y=[umap_2],
            mode="markers",
            marker_line_color="midnightblue",
            marker_color="lightskyblue",
            marker_line_width=2,
            marker_size=15,
            marker_symbol="star-diamond",
            name="Highlighted Patient",  # 设置名称
        )
    )

    # 隐藏坐标轴
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    # 响应式布局，设置画布颜色为白色，图例横向显示
    fig.update_layout(
        autosize=True,  # 开启自动调整大小
        title={
            "text": "UMAP Projection with NMF Clustering",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        showlegend=True,
        paper_bgcolor="white",  # 设置画布背景颜色为白色
        plot_bgcolor="white",  # 设置绘图区域背景为白色
        legend=dict(
            orientation="h",  # 设置图例为横向排列
            x=0.5,  # 图例居中
            xanchor="center",
            y=-0.15,  # 图例的位置调整
            yanchor="bottom",
            bgcolor="rgba(255, 255, 255, 0.7)",  # 图例背景颜色，半透明
            bordercolor="gray",  # 图例边框颜色
            borderwidth=1,  # 图例边框宽度
        ),
    )

    return fig, predicted_topic, score