from pygwalker.api.gradio import PYGWALKER_ROUTE
import gradio as gr
import torch
from model.model import MimicTransformer
from model.setup_func import get_model_results, find_related_summaries, extract_info, score_topic_for_text
from model.plot_func import show_pygwalker, plot_umap
from utils.utils import MODEL_LIST, visualize_attn
from transformers import AutoTokenizer, AutoModel, set_seed, pipeline
import pandas as pd
import re
from zhipuai import ZhipuAI
import os
import language_tool_python

# 设置代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:33210"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:33210"

torch.manual_seed(0)
set_seed(34)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

df = pd.read_csv(r"data/csv/PMC-Patients.csv")
plot_df = pd.read_csv(r"data/csv/nmf_umap_clusters.csv")

model_path = MODEL_LIST["marking"]["model_path"]

# Model initialization
max_width = 55  # 设定统一的总宽度
divider = "*" * 20 + " MODEL_INIT " + "*" * 20
print(divider.center(max_width))

# Initialize Mimic model
print("----> Mimic Init <----".center(max_width))
mimic = MimicTransformer(
    tokenizer_name=MODEL_LIST["mimic"]["model_url"],
    cutoff=512,
    model_path=MODEL_LIST["mimic"]["model_path"],
)
mimic_tokenizer = mimic.tokenizer
mimic.eval()

# Initialize similarity model
print("----> Similarity Init <----".center(max_width))
similarity_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_LIST["similarity"]["model_path"]
)
similarity_model = AutoModel.from_pretrained(MODEL_LIST["similarity"]["model_path"])
similarity_model.eval()

# Initialize NER pipeline
print("----> NER Init <----".center(max_width))
pipe = pipeline("token-classification", model=MODEL_LIST["ner"]["model_url"])

# Load related data
print("----> Loading Related Data <----".center(max_width))
related_tensor = torch.load(MODEL_LIST["similarity"]["embedding_file_path"])
all_summaries = pd.read_csv(MODEL_LIST["similarity"]["csv_file_path"])[
    "patient"
].to_list()

print("----> ZhipuAI Init <----".center(max_width))
client = ZhipuAI(api_key=MODEL_LIST["zhipu"]["api-key"])

print(("*" * 18 + " END_MODEL_INIT " + "*" * 18).center(max_width))


def run(text):
    torch.manual_seed(0)
    set_seed(34)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    model_results = get_model_results(mimic, mimic_tokenizer, pipe, text)

    related_summaries = find_related_summaries(
        similarity_model, similarity_tokenizer, text, related_tensor, all_summaries
    )
    plot_p, p_topic, p_score = plot_umap(plot_df, text, model_path)

    return (
        visualize_attn(model_results=model_results),
        gr.update(value=related_summaries, visible=True),
        plot_p,
        p_topic,
        p_score,
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True)

    )


# 定义根据 patient_uid 查找对应行的函数
def find_patient_row(patient_uid, df):
    # 查找 patient_uid 匹配的行
    patient_row = df[df['patient_uid'] == patient_uid]

    # 如果找到了匹配的行，返回该行的 DataFrame
    if not patient_row.empty:
        return patient_row  # 直接返回 DataFrame 格式
    else:
        # 如果找不到，返回一个空的 DataFrame
        return pd.DataFrame(columns=df.columns)


def highlight(val, column):
    # 如果是 'gender' 列且值不是 'M' 或 'F'，则高亮显示
    if column == 'gender' and val not in ['M', 'F']:
        return 'background-color: #FF6347; color: #FFFFFF; font-weight: bold; border: 2px solid #FF4500'

    # 如果是 'age' 列，检查其是否为空列表或包含无效值
    if column == 'age':
        # 检查 age 列是空列表或包含无效的值
        if not val or all(v is None for v in val):  # 如果是空列表或包含所有 None 的列表
            return 'background-color: #FF6347; color: #FFFFFF; font-weight: bold; border: 2px solid #FF4500'

    return ''


def highlight_invalid_gender(df):
    # 对 'gender' 和 'age' 列分别应用高亮
    return df.style.map(lambda val: highlight(val, 'gender'), subset=['gender']) \
        .map(lambda val: highlight(val, 'age'), subset=['age'])


def on_select_value(selected_row):
    # 选择的行是一个 DataFrame，所以我们从中提取 patient 列的值
    if selected_row is not None and not selected_row.empty:
        patient_value = selected_row.iloc[0]["patient"]  # 获取 patient 列的值
        return patient_value  # 返回该值到文本框
    return ""  # 如果没有选择任何行，返回空字符串


def lda_display_html():
    with open("Patient_lda_visualization.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return html_content


# 当 patient_uid 被提交时，查询对应的 patient 信息
def on_patient_uid_submit(patient_uid):
    try:
        # 检查 patient_uid 格式
        if not re.match(r'^\d{7}-\d{1,3}$', patient_uid):
            return "Error UID Format，e.g. 7665777-1", pd.DataFrame(columns=df.columns)

        # 查找对应行
        patient_info = find_patient_row(patient_uid, df)

        # 如果未找到匹配的行
        if patient_info.empty:
            return "Cannot find the data，plz check the UID correction", pd.DataFrame(columns=df.columns)

        # 对 gender 列进行高亮显示
        styled_patient_info = highlight_invalid_gender(patient_info)

        return None, styled_patient_info

    except Exception as e:
        # 捕获所有异常并返回错误信息到 message_output
        error_message = f"Fatal Error: {str(e)}"
        return error_message, pd.DataFrame(columns=df.columns)


def extract_info_row(patient):
    error_list = []  # 用于存储错误信息
    suggestion_list = []  # 用于存储建议信息
    tool = language_tool_python.LanguageTool('en-US')
    s = """"""
    error_count = 0
    try:
        # 尝试从 extract_info 中提取信息
        info = extract_info(client, patient)  # 示例信息提取
        matches = tool.check(patient)
        age = info.get("age", [])
        gender = info.get("gender", "Unknown")
        predicted_topic, score, nmf_2d = score_topic_for_text(model_path, patient)

        # 错误捕获与处理
        if not age:
            s += '[Age Error]\t' + '\nMessage: ' + 'Cannot find age information' + '\nSuggestion: ' + 'Plz check if there is age information in the context\n\n'
            error_count += 20
        if gender == "Unknown":
            s += '[Gender Error]\t' + '\nMessage: ' + 'Gender is unknown' + '\nSuggestion: ' + 'There is not gender information in the context\n\n'
            error_count += 20
        for match in matches:
            if match.message:
                s += '[Spelling Error]\t' + '\nMessage: {}'.format(match.message)
                error_count += 1
            if match.replacements:
                s += '\nSuggestion: {}'.format('; '.join(match.replacements[:5]))
            s += '\n{}\n{}'.format(
                match.context, ' ' * match.offsetInContext + '\n\n'
            )
            print(s)

    except Exception as e:
        # 如果提取信息时发生异常，记录异常信息
        s += '[Fatal Error]\t' + 'Message: ' + f"Message box error {e} " + 'Plz check your upload content\n'

        age = []
        gender = "Unknown"

    # 创建 DataFrame
    df = pd.DataFrame([{"age": age, "gender": gender}])
    df['gender'] = df['gender'].apply(str)
    df_styled = highlight_invalid_gender(df)

    # 判断是否显示绿色按钮
    if not s:
        button_visibility = gr.update(visible=True)  # 显示绿色按钮
        error_visibility = gr.update(visible=False)

    else:
        button_visibility = gr.update(visible=False)  # 不显示绿色按钮
        error_visibility = gr.update(visible=True)
    save_info = gr.update(visible=False)
    score = score - error_count

    # 返回错误信息、建议和 DataFrame
    return s, df_styled, button_visibility, error_visibility, score, save_info


def save(text):
    predicted_topic, score, nmf_2d = score_topic_for_text(model_path, text)
    plot_df.loc[len(plot_df)] = [nmf_2d[0, 0], nmf_2d[0, 1], predicted_topic]
    button_visibility = gr.update(visible=True)
    # 保存 DataFrame 到指定路径的 CSV 文件
    plot_df.to_csv("./data/csv/nmf_umap_clusters.csv", index=False)
    return button_visibility


# main ui
with gr.Blocks(css=MODEL_LIST["css"], js=MODEL_LIST["js"]) as demo:
    with gr.Tab("DataViewer"):  # Page 1 标签
        gr.HTML(show_pygwalker(df))  # 展示 pygwalker 可视化

    with gr.Tab("ContentDetail"):  # Page 2 标签
        gr.Markdown("""
                        # NLP Medical System
                        🤖This system is a medical information extraction and visualization system based on natural language processing (NLP) and machine learning, which aims to help medical professionals and researchers extract key information from patients' hospitalization summaries and provide intelligent similarity analysis and visualization functions.
                        """)

        with gr.Row():
            with gr.Column():
                with gr.Column(visible=False) as Patient_uid_part:
                    patient_uid_input = gr.Textbox(
                        label="Enter Patient UID",
                        placeholder="please enter correct uid format e.g. 7665777-1",
                        interactive=True,
                    )
                    message_output = gr.Textbox(
                        label="Message",
                        interactive=False,
                        lines=1,
                        max_lines=1,
                    )
                    patient_info_output = gr.DataFrame(label="Patient Info", interactive=False, visible=True,
                                                       elem_id="patient_info_output")
                with gr.Row():
                    uid_show_button = gr.Button("Show Patient UID Form")
                    uid_show_button.click(
                        lambda: gr.update(visible=True),
                        inputs=None,
                        outputs=Patient_uid_part
                    )
                    uid_hide_button = gr.Button("Hide Patient UID Form")
                    uid_hide_button.click(
                        lambda: gr.update(visible=False),
                        inputs=None,
                        outputs=Patient_uid_part
                    )

            with gr.Column():
                with gr.Column(visible=False) as New_patient:
                    patient_new_patient = gr.Textbox(
                        label="Enter Patient Contents",
                        placeholder="please enter new patient content",
                        interactive=True,
                    )
                    with gr.Column():
                        error_info = gr.Textbox(
                            label="Error",
                            interactive=False,
                            # lines=1,
                            # max_lines=1,

                        )
                        save_button = gr.Button("Save", visible=False)
                        save_info = gr.Markdown("""Save Sucess""", visible=False)
                    with gr.Row():
                        score_p = gr.Textbox(
                            label="Scores",
                            interactive=False,
                        )
                        new_patient_info = gr.DataFrame(label="Patient Info", interactive=False, visible=True)

                    patient_new_patient.submit(
                        extract_info_row,
                        inputs=patient_new_patient,
                        outputs=[error_info, new_patient_info, save_button, error_info, score_p, save_info]
                    )
                with gr.Row():
                    # 隐藏按钮
                    patient_show_button = gr.Button("Show Patient Form")
                    patient_show_button.click(
                        lambda: gr.update(visible=True),
                        inputs=None,
                        outputs=New_patient
                    )
                    patient_hide_button = gr.Button("Hide Patient Form")
                    patient_hide_button.click(
                        lambda: gr.update(visible=False),
                        inputs=None,
                        outputs=New_patient
                    )

        with gr.Row():
            patient_uid_input.submit(
                on_patient_uid_submit,
                inputs=patient_uid_input,
                outputs=[message_output, patient_info_output],
            )

        with gr.Row() as row:
            input = gr.Textbox(
                label="Input Discharge Summary Here",
                placeholder="sample discharge summary",
                text_align="left",
                interactive=True,
            )
            patient_info_output.select(
                on_select_value,  # 回调函数
                inputs=patient_info_output,  # DataFrame 作为输入源
                outputs=input,  # 文本框作为输出目标
            )

        with gr.Row() as row:
            btn = gr.Button(value="Submit")
        with gr.Row() as row:
            attn_viz = gr.HTML()

        with gr.Row() as row:
            with gr.Row() as row:
                patient_score = gr.Textbox(
                    label="Patient Scores",
                    placeholder="will show the score of selected patient",
                    interactive=False,
                )
                patient_cluster = gr.Textbox(
                    label="Patient Cluster",
                    placeholder="will show the cluster of selected patient",
                    interactive=False,
                )
            related = gr.DataFrame(
                value=None,
                headers=["Similarity Score", "Related Discharge Summary"],
                row_count=5,
                datatype=["number", "str"],
                col_count=(2, "fixed"),
                visible=False,
            )

        with gr.Row():
            with gr.Column(scale=1):  # 左边区域，放置图表
                plot_output = gr.Plot(label="UMAP Visualization")

            with gr.Column(scale=1):  # 右边区域，放置 HTML 文件
                pass

        # initial run
        btn.click(run, inputs=[input], outputs=[attn_viz, related, plot_output, patient_cluster, patient_score])
        save_button.click(save, inputs=[patient_new_patient], outputs=[save_info])

# 启动应用，包含 pygwalker 路由
app = demo.launch(app_kwargs={"routes": [PYGWALKER_ROUTE]})
