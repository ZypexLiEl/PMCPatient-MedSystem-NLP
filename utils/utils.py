import urllib.request, urllib.error, urllib.parse
import json
import pandas as pd
import ssl
import torch
import re
import difflib
from pprint import pprint
from captum.attr import visualization


MODEL_LIST = {
    "mimic": {
        "model_path": r"data/ckpt/checkpoint_0_9113.bin",
        "model_url": r"data/ckpt/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/model",
    },
    "similarity": {
        "model_path": r"data/ckpt/models--kamalkraj--BioSimCSE-BioLinkBERT-BASE/snapshots/model",
        "embedding_file_path": r"data/ckpt/embeddings_cpu.pt",
        "csv_file_path": r"data/csv/cleaned_patients.csv",
    },
    "ner": {
        "model_url": r"data/ckpt/models--alvaroalon2--biobert_diseases_ner/snapshots/model",
    },
    "zhipu": {"api-key": "82cae7677d1ae318ea03f8fb5b2a73cd.z27ECVAFglgOeF5i"},
    "marking": {
        "model_path": r"data/ckpt"
    },
    "css": """
    #patient_info_output {
        width: 600px;  /* 设置表格宽度 */
        max-width: 100%; /* 限制最大宽度 */
        overflow-x: auto;  /* 超出宽度时显示水平滚动条 */
    }
    #patient_info_output td {
        word-wrap: break-word;  /* 自动换行 */
        max-width: 200px;  /* 设置列的最大宽度 */
    }
    """,
    "js": """
        function createGradioAnimation() {
            var container = document.createElement('div');
            container.id = 'gradio-animation';
            container.style.fontSize = '2em';
            container.style.fontWeight = 'bold';
            container.style.textAlign = 'center';
            container.style.marginBottom = '20px';

            var text = '🧐☝Welcome to PMC-Medsystem!🤪🫵';
            for (var i = 0; i < text.length; i++) {
                (function(i){
                    setTimeout(function(){
                        var letter = document.createElement('span');
                        letter.style.opacity = '0';
                        letter.style.transition = 'opacity 0.5s';
                        letter.innerText = text[i];

                        container.appendChild(letter);

                        setTimeout(function() {
                            letter.style.opacity = '1';
                        }, 50);
                    }, i * 250);
                })(i);
            }

            var gradioContainer = document.querySelector('.gradio-container');
            gradioContainer.insertBefore(container, gradioContainer.firstChild);

            return 'Animation created';
        }
    """,
}


REST_URL = "http://data.bioontology.org"
API_KEY = ""
ssl._create_default_https_context = ssl._create_unverified_context


class PyTMinMaxScalerVectorized(object):
    """
    From https://discuss.pytorch.org/t/using-scikit-learns-scalers-for-torchvision/53455
    Transforms each channel to the range [0, 1].
    """

    def __call__(self, tensor):
        scale = 1.0 / (
            tensor.max(dim=0, keepdim=True)[0] - tensor.min(dim=0, keepdim=True)[0]
        )
        tensor.mul_(scale).sub_(tensor.min(dim=0, keepdim=True)[0])
        return tensor


def _normalized_diseases(text_list, disease):
    candidates = difflib.get_close_matches(disease, text_list)
    if len(candidates) > 0:
        return candidates[0]
    return ""


def clean_disease_string(disease):
    disease = disease.strip().lower()
    disease = re.sub(r"[^\w\s]", "", disease)
    return disease


def normalized_diseases(text, disease_list):
    disease_list = list(set(disease_list))
    text_split = text.split()
    normalized = []
    for disease in disease_list:
        # case when the disease is one word
        if " " not in disease:
            candidate = _normalized_diseases(disease=disease, text_list=text_split)
            if len(candidate) > 0:
                candidate = clean_disease_string(candidate)
                normalized.append(candidate)
        else:
            concept = ""
            for disease_word in disease.split():
                candidate = _normalized_diseases(
                    text_list=text_split, disease=disease_word
                )
                if len(candidate) > 0:
                    concept += candidate + " "
            if len(concept.split()) == len(disease.split()):
                concept = clean_disease_string(concept)
                normalized.append(concept)
    return list(set(normalized))


def get_diseases(text, pipe):
    results = pipe(text)
    diseases = []
    disease_span = []
    for result in results:
        ent = result["entity"]
        # start of a new entity
        if ent == "B-DISEASE":
            disease_span = result["start"], result["end"]
        elif ent == "I-DISEASE":
            if len(disease_span) == 0:
                disease_span = []
            else:
                disease_span = disease_span[0], result["end"]
        else:
            if len(disease_span) > 1:
                disease = text[disease_span[0] : disease_span[1]]
                if len(disease) > 2:
                    diseases.append(disease)
            disease_span = []
    if len(disease_span) > 1:
        disease = text[disease_span[0] : disease_span[1]]
        diseases.append(disease)
    normalized = normalized_diseases(text, diseases)
    return normalized


def find_end(text):
    """Find the end of the report."""
    ends = [len(text)]
    patterns = [
        re.compile(r"BY ELECTRONICALLY SIGNING THIS REPORT", re.I),
        re.compile(r"\n {3,}DR.", re.I),
        re.compile(r"[ ]{1,}RADLINE ", re.I),
        re.compile(r".*electronically signed on", re.I),
        re.compile(r"M\[0KM\[0KM"),
    ]
    for pattern in patterns:
        matchobj = pattern.search(text)
        if matchobj:
            ends.append(matchobj.start())
    return min(ends)


def pattern_repl(matchobj):
    """
    Return a replacement string to be used for match object
    """
    return " ".rjust(len(matchobj.group(0)))


def clean_text(text):
    """
    Clean text
    """

    # Replace [**Patterns**] with spaces.
    text = re.sub(r"\[\*\*.*?\*\*\]", pattern_repl, text)
    # Replace `_` with spaces.
    text = re.sub(r"_", " ", text)

    start = 0
    end = find_end(text)
    new_text = ""
    if start > 0:
        new_text += " " * start
    new_text = text[start:end]

    # make sure the new text has the same length of old text.
    if len(text) - end > 0:
        new_text += " " * (len(text) - end)
    return new_text


def get_drg_link(drg_code):
    drg_code = str(drg_code)
    if len(drg_code) == 1:
        drg_code = "00" + drg_code
    elif len(drg_code) == 2:
        drg_code = "0" + drg_code
    return f"https://www.findacode.com/code.php?set=DRG&c={drg_code}"


def prettify(dict_list, k):
    li = [di[k] for di in dict_list]
    result = "\n".join(l for l in li)
    return result


def get_json(text_to_annotate):
    url = (
        REST_URL
        + "/annotator?text="
        + urllib.parse.quote(text_to_annotate)
        + "&ontologies=ICD9CM"
        + "&longest_only=false"
        + "&exclude_numbers=false"
        + "&whole_word_only=true"
        + "&exclude_synonyms=false"
    )
    opener = urllib.request.build_opener()
    opener.addheaders = [("Authorization", "apikey token=" + API_KEY)]
    try:
        return json.loads(opener.open(url).read())
    except:
        return []


def parse_results(results):
    if len(results) == 0:
        return []
    rlist = []
    for result in results:
        annotations = result["annotations"]
        for annotation in annotations:
            start = annotation["from"] - 1
            end = annotation["to"] - 1
            text = annotation["text"]
            rlist.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "link": result["annotatedClass"]["@id"],
                }
            )
    return rlist


def get_icd_annotations(text):
    response = get_json(text)
    annotation_list = parse_results(response)
    return annotation_list


def subfinder(mylist, pattern):
    mylist = mylist.tolist()
    pattern = pattern.tolist()
    return list(filter(lambda x: x in pattern, mylist))


def tokenize_icds(tokenizer, annotations, token_ids):
    icd_tokens = torch.zeros(token_ids.shape)
    for annotation in annotations:
        icd = annotation["text"]
        icd_token_ids = tokenizer(
            icd, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]
        # find index of the beginning icd token
        starting_indices = (token_ids == icd_token_ids[0]).nonzero(as_tuple=False)
        num_icd_tokens = icd_token_ids.shape[0]

        # if there's more than 1 icd token for the given annotation
        if num_icd_tokens > 1:
            # if there's only one starting index
            if starting_indices.shape[0] == 1:
                starting_index = starting_indices.item()
                icd_tokens[starting_index : starting_index + num_icd_tokens] = 1
            # if there's more than 1 starting index, determine which is the appropriate
            else:
                for starting_index in starting_indices:
                    if token_ids[starting_index + num_icd_tokens] == icd_token_ids:
                        icd_tokens[starting_index : starting_index + num_icd_tokens] = 1

        # otherwise, set the corresponding index to a value of 1
        else:
            icd_tokens[starting_indices] = 1
    return icd_tokens


def get_attribution(text, tokenizer, model_outputs, inputs, k=7):
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    padding_idx = 512
    if "[PAD]" in tokens:
        padding_idx = tokens.index("[PAD]")
    tokens = tokens[:padding_idx][1:-1]
    attn = model_outputs[-1][0]
    agg_attn, final_text = reconstruct_text(
        tokenizer=tokenizer, tokens=tokens, attn=attn
    )
    return agg_attn, final_text


def reconstruct_text(tokenizer, tokens, attn):
    """
    find a word -> token_id mapping that allows you to
    perform an aggregation on the sub-tokens' attention
    values
    """
    reconstructed_text = tokenizer.convert_tokens_to_string(tokens)
    num_subtokens = len([t for t in tokens if t.startswith("#")])
    aggregated_attn = torch.zeros(len(tokens) - num_subtokens)
    token_indices = [0]
    token_idx = 0
    reconstructed_tokens = []
    for i, token in enumerate(tokens[1:], start=1):
        # case when a token is a subtoken
        if token.startswith("#"):
            token_indices.append(i)
        else:
            # reconstruct the tokens to make sure you're doing this correctly
            reconstructed_token = "".join(
                tokens[i].replace("#", "") for i in token_indices
            )
            reconstructed_tokens.append(reconstructed_token)
            # find the corresponding attention vectors
            aggregated_attn[token_idx] = torch.mean(attn[token_indices])
            # create new index list
            token_indices = [i]
            token_idx += 1
    # reconstruct the tokens to make sure you're doing this correctly
    reconstructed_token = "".join(tokens[i].replace("#", "") for i in token_indices)
    reconstructed_tokens.append(reconstructed_token)
    # find the corresponding attention vectors
    aggregated_attn[token_idx] = torch.mean(attn[token_indices])

    # final representation of text
    final_text = " ".join(reconstructed_tokens).replace(" .", ".")
    final_text = final_text.replace(" ,", ",")
    # final_text == reconstructed_text
    return aggregated_attn, reconstructed_tokens


def load_rule(path):
    rule_df = pd.read_csv(path)

    # remove MDC 15 - neonate and couple other codes related to postcare
    if "MS" in path:
        msk = (rule_df["MDC"] != "15") & (
            ~rule_df["MS-DRG"].isin([945, 946, 949, 950, 998, 999])
        )
        space = sorted(rule_df[msk]["DRG_CODE"].unique())
    elif "APR" in path:
        msk = (rule_df["MDC"] != "15") & (~rule_df["APR-DRG"].isin([860, 863]))
        space = sorted(rule_df[msk]["DRG_CODE"].unique())

    drg2idx = {}
    for d in space:
        drg2idx[d] = len(drg2idx)
    i2d = {v: k for k, v in drg2idx.items()}

    d2mdc, d2w = {}, {}
    for _, r in rule_df.iterrows():
        drg = r["DRG_CODE"]
        mdc = r["MDC"]
        w = r["WEIGHT"]
        d2mdc[drg] = mdc
        d2w[drg] = w

    return rule_df, drg2idx, i2d, d2mdc, d2w


def visualize_attn(model_results):
    class_id = model_results["class_dsc"]
    prob = model_results["prob"]
    attn = model_results["attn"]
    tokens = model_results["tokens"]
    scaler = PyTMinMaxScalerVectorized()
    normalized_attn = scaler(attn)
    viz_record = visualization.VisualizationDataRecord(
        word_attributions=normalized_attn,
        pred_prob=prob,
        pred_class=class_id,
        true_class=class_id,
        attr_class=0,
        attr_score=1,
        raw_input_ids=tokens,
        convergence_score=1,
    )
    return visualize_text(
        viz_record,
        drg_link=model_results["drg_link"],
        icd_annotations=model_results["icd_results"],
        diseases=model_results["diseases"],
    )


def modify_attn_html(attn_html):
    attn_split = attn_html.split("<mark")
    htmls = [attn_split[0]]
    for html in attn_split[1:]:
        # wrap around href tag
        href_html = f'<a href="https://" \
            <mark{html} \
            </a>'
        htmls.append(href_html)
    return "".join(htmls)


def modify_code_html(html, link, icd=False):
    html = html.split("<td>")[1].split("</td>")[0]
    href_html = f'<td><a href="{link}"{html}</a></td>'
    if icd:
        href_html = href_html.replace("<td>", "").replace("</td>", "")
    return href_html


def modify_drg_html(html, drg_link):
    return modify_code_html(html=html, link=drg_link, icd=False)


def get_icd_html(icd_list):
    if len(icd_list) == 0:
        return '<td><text style="padding-left:2em"><b>N/A</b></text></td>'
    final_html = "<td>"
    icd_set = set()
    style = "border-style: solid; overflow: visible; min-width: calc(min(0px, 100%)); border-width: var(--block-border-width);"
    for i, icd_dict in enumerate(icd_list):
        text, link = icd_dict["text"], icd_dict["link"]
        if text in icd_set:
            continue
        # tmp_html = visualization.format_classname(classname=text)
        # html = modify_code_html(html=tmp_html, link=link, icd=True)
        # style="padding-left:2em; font-weight:bold;"
        icd_set.add(text)
        if i + 1 < len(icd_list):
            text += ","
        html = f'<a style="{style}" href="{link}">{text}</a><br>'
        final_html += html
    return final_html + "</td>"


def get_disease_html(diseases):
    if len(diseases) == 0:
        return '<td><text style="padding-left:2em"><b>N/A</b></text></td>'
    diseases = list(set(diseases))
    diseases_str = ", ".join(diseases)
    html = visualization.format_classname(classname=diseases_str)
    return html


# copied out of captum because we need raw html instead of a jupyter widget
def visualize_text(datarecord, drg_link, icd_annotations, diseases):
    dom = ["<table width: 100%>"]
    rows = [
        "<th style='text-align: left'>Predicted DRG</th>"
        "<th style='text-align: left'>Word Importance</th>"
        "<th style='text-align: left'>Diseases</th>"
        "<th style='text-align: left'>ICD Concepts</th>"
    ]
    pred_class_html = visualization.format_classname(datarecord.pred_class)
    icd_class_html = get_icd_html(icd_annotations)
    disease_html = get_disease_html(diseases)
    pred_class_html = modify_drg_html(html=pred_class_html, drg_link=drg_link)
    word_attn_html = visualization.format_word_importances(
        datarecord.raw_input_ids, datarecord.word_attributions
    )
    rows.append(
        "".join(
            [
                "<tr>",
                pred_class_html,
                word_attn_html,
                disease_html,
                icd_class_html,
                "<tr>",
            ]
        )
    )

    dom.append("".join(rows))
    dom.append("</table>")
    html = "".join(dom)

    return html
