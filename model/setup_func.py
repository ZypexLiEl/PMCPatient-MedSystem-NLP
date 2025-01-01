import torch
from utils.utils import (
    load_rule,
    get_diseases,
    get_drg_link,
    get_icd_annotations,
    get_attribution,
    clean_text,
)
from sklearn.metrics.pairwise import cosine_similarity
import json
import joblib
import numpy as np
import os


def get_model_results(mimic, mimic_tokenizer, pipe, text: str):
    rule_df, _, i2d, _, _ = load_rule("data/csv/MSDRG_RULE13.csv")

    # Step 1: Clean and tokenize text
    text = clean_text(text)
    inputs = mimic_tokenizer(
        text, return_tensors="pt", padding="max_length", max_length=512, truncation=True
    )

    # Step 2: Get model outputs
    with torch.no_grad():
        outputs = mimic(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            drg_labels=None,
        )

    # Step 3: Attribution and reconstructed text
    attribution, reconstructed_text = get_attribution(
        text=text, tokenizer=mimic_tokenizer, model_outputs=outputs, inputs=inputs, k=10
    )

    # Step 4: Extract logits and DRG classification
    logits = outputs[0][0]
    out = logits.detach().cpu()[0]
    drg_code = i2d[out.argmax().item()]
    prob = torch.nn.functional.softmax(out).max()

    # Step 5: Extract diseases using NER pipeline
    diseases = get_diseases(text=text, pipe=pipe)

    # Step 6: Retrieve DRG link and ICD annotations
    drg_link = get_drg_link(drg_code=drg_code)
    icd_results = get_icd_annotations(text=text)

    # Step 7: Retrieve DRG description
    row = rule_df[rule_df["DRG_CODE"] == drg_code]
    drg_description = row["DESCRIPTION"].values[0]

    # Step 8: Compile results
    return {
        "class": drg_code,
        "prob": prob,
        "attn": attribution,
        "tokens": reconstructed_text,
        "logits": logits,
        "diseases": diseases,
        "drg_link": drg_link,
        "icd_results": icd_results,
        "class_dsc": drg_description,
    }


def find_related_summaries(
    similarity_model,
    similarity_tokenizer,
    text,
    related_tensor,
    all_summaries: list,
    top_k=5,
):
    # 将输入文本编码到模型
    inputs = similarity_tokenizer(
        text, padding="max_length", truncation=True, return_tensors="pt", max_length=512
    )

    # 获取模型输出
    with torch.no_grad():
        outputs = similarity_model(**inputs)

    # 获取 last_hidden_state 的平均值作为句子嵌入
    hidden_states = outputs.last_hidden_state
    query_embedding = (
        hidden_states.mean(dim=1).squeeze().cpu().numpy()
    )  # 在tokens维度上取均值

    # 计算输入文本与所有摘要的余弦相似度
    scores = cosine_similarity([query_embedding], related_tensor).flatten()

    # 找到最高相似度的前 top_k 条
    topk_indices = scores.argsort()[-top_k-1:][::-1]  # 获取得分最高的索引，降序排列
    topk_indices = topk_indices[1:]
    topk_scores = scores[topk_indices]
    print(topk_indices, topk_scores)

    # 生成结果列表
    summary_score_list = []
    for idx, score in zip(topk_indices, topk_scores):
        corresp_summary = all_summaries[idx]
        summary_score_list.append([round(score, 2), corresp_summary])

    return summary_score_list


def extract_info(client, text):
    # 设计 prompt，要求提取时间和性别
    prompt = f"""
    Please extract and format the following information from the input text:

    1. **Age Information**: Identify all mentions of age in the text, where age is specified as a number followed by a time unit (e.g., "six-year-old", "27 year old"). Extract each instance and format it as a list in the structure [{{float}}, {{str}}], where {{float}} is the numeric age and {{str}} is one of the following units: {{'month', 'hour', 'year', 'week', 'day'}}.

    For example:
    - "six-year-old" -> [[6.0, 'year']]
    - "27 year old" -> [[27.0, 'year']]
    - "74-year old" -> [[74.0, 'year']]

    If multiple matches are found, include all of them in a list.

    2. **Gender Information**: Identify the inferred gender of the individual in the text. Return "M" if the text includes references to male-associated terms (e.g., "man", "male", "he", "prostate"), and return "F" if the text includes female-associated terms (e.g., "woman", "female", "she", "ovary"). If no clear gender is inferred, return "Unknown".

    DON'T RETURN IN THE FORM OF A BLOCK OF CODE,Return the output in THE FOLLOWING FORMAT(with no explanation):
    {{
        "age": List[List[float, str]],
        "gender": str
    }}

    Input text:
    "{text[:100]}"
    """

    # 调用 API 进行推理
    response = client.chat.completions.create(
        model="glm-4-plus",  # 请填写您要调用的模型名称
        messages=[{"role": "user", "content": prompt}],
    )

    # 从返回的 response 中提取结果
    result = response.choices[0].message.content
    result = result.strip()
    print(result)

    try:
        parsed_json = json.loads(result)  # 解析为JSON对象
        return parsed_json
    except Exception as e:
        print("No JSON found.")
        return None


# Softmax with temperature function
def softmax_with_temperature(x, temperature=1.0):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()


# 计算 Softmax 得分的函数
def score_softmax(softmax_output, alpha=0.5, beta=0.5):
    eps = 1e-12
    softmax_output = np.array(softmax_output)
    H = -np.sum(softmax_output * np.log(softmax_output + eps))
    H_norm = H / np.log(len(softmax_output))

    sorted_probs = np.sort(softmax_output)[::-1]
    p_max, p_second = sorted_probs[0], sorted_probs[1]
    C = 1 - (p_second / (p_max + eps))

    score = alpha * (1 - H_norm) + beta * C

    min_score = 0
    max_score = 1
    scaled_score = 100 * (score - min_score) / (max_score - min_score)

    threshold = 1e-6
    if scaled_score < threshold:
        scaled_score = 0

    return scaled_score


# 主函数，加载模型，计算主题概率并评分
def score_topic_for_text(model_path, text, n_topics=10, temperature=0.01):
    # print(os.getcwd())
    # 读取已保存的模型
    nmf_model = joblib.load(os.path.join(model_path, "nmf_model.pkl"))
    vectorizer = joblib.load(os.path.join(model_path, "vectorizer.pkl"))
    umap = joblib.load(os.path.join(model_path, "umap_model.pkl"))

    # 文本向量化
    new_text_tfidf = vectorizer.transform([text])

    # 使用已训练的 NMF 模型获取主题分布
    new_text_topics = nmf_model.transform(new_text_tfidf)

    # 对主题分布应用 Softmax 转换为概率分布
    topic_probabilities = softmax_with_temperature(new_text_topics[0], temperature)

    # 找到概率最大的主题
    predicted_topic = np.argmax(topic_probabilities)

    # 计算 Softmax 得分
    score = score_softmax(topic_probabilities)

    nmf_2d = umap.transform(new_text_topics)

    return predicted_topic, score, nmf_2d