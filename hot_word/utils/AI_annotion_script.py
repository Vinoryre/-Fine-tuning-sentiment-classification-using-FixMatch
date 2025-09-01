import pandas as pd
from tqdm import tqdm
import requests
import os

# TODO: change your dataset path...
CSV_FILE = "../Dataset/chat_data_set/df_0218_all_0205_final.csv"
INDEX_FILE = "../utils/AI_annotation_processed_indices.csv"
OUTPUT_FILE = "../Dataset/AI_annotation_data/ai_0205_final.csv"
BATCH_SIZE = 30000
RESET = False


# 加载已处理索引函数
def load_processed_indices():
    if os.path.exists(INDEX_FILE):
        return pd.read_csv(INDEX_FILE)['index'].tolist()
    return []


# 保存已处理的索引函数
def save_processed_indices(indices):
    pd.DataFrame({'index': indices}).to_csv(INDEX_FILE, index=False)


# 取下一批样本函数
def get_next_batch():
    df = pd.read_csv(CSV_FILE)
    processed = load_processed_indices()

    all_indices = set(df.index.tolist())
    remaining_indices = list(all_indices - set(processed))

    if not remaining_indices:
        print("✅ 所有样本都已处理完。")
        return None

    # 从未处理的索引中取下一批
    next_batch_indices = remaining_indices[:BATCH_SIZE]
    batch_df = df.loc[next_batch_indices]

    print(f"📦 本次取出样本数：{len(batch_df)}")
    return batch_df, processed


def reset_progress():
    """清除进度文件"""
    for file in [INDEX_FILE, OUTPUT_FILE]:
        if os.path.exists(file):
            os.remove(file)
            print(f"🔄 已删除 {file}")
    print("✅ 进度已清除。")


# 调用AI模型接口函数
def request_to_model(query_str, is_classify=False, message=None, temperature=0.5, top_p=0.5):
    """
    调用glm-4-9b-chat模型
    :param query_str: 输入进去的文本内容
    :param is_classify:
    :param message:
    :param temperature:
    :param top_p:
    :return: 返回模型根据文本给出的输出
    """
    if message is not None:
        data_config = {
            "model": "glm-4-9b-chat",
            "messages": message,
            "temperature": temperature,
            "top-p": top_p,
            "is_classify": is_classify
        }
    else:
        data_config = {
            "model": "glm-4-9b-chat",
            "messages": query_str,
            "temperature": temperature,
            "top_p": top_p,
            # "max_length" : 4096,
            "is_classify": is_classify
        }

    req = requests.post(
        '',  # TODO: replace your glm key
        json=data_config
    )
    x = req.json()
    return x.get('choices')[0]['message']['content']


# GLM AI 提示词版标注数据
def glm_emotion_annotate(csv_file):
    """
    自动化标注csv里面的文本标签数据
    :param csv_file:
    :return: 返回一个新的CSV，带有AI标注的标签
    """

    # 读取数据
    df_raw, processed = get_next_batch()

    # 小批次遍历df_raw
    batch_size = 5
    num_batches = (len(df_raw) + batch_size - 1) // batch_size

    # 映射表
    label_map = {'中性': int(2), '负向': int(1), '正向': int(0), '未知': None}

    # --生成器
    def batch_iter(df, b_s):
        for i in range(0, len(df), b_s):
            yield df.iloc[i:i + b_s]

    # 启动前先加载已有标注结果
    if os.path.exists(OUTPUT_FILE):
        df_ai = pd.read_csv(OUTPUT_FILE)
    else:
        df_ai = pd.DataFrame(columns=['text', 'label'])

    for batch_df in tqdm(batch_iter(df_raw, batch_size), total=num_batches, desc='Processing batches'):
        # 开始处理每一个批次的df逻辑
        pharse = "\n".join([f"{i + 1}. {txt}" for i, txt in enumerate(batch_df['text'])])

        query_str = f"""你是一名资深的游戏制作主管,深谙中国玩家在游戏社区,聊天中表达情绪的方式,例如中文玩家常通过含蓄或反讽表达情绪，不要只看字面意思,对混合情绪,以主要情感为准.
        你的任务是阅读我提供的聊天文本,并根据整体语气只能将其划分为以下三类情感其中之一，你必须做出选择只能划分为其中一个类别，不能出现中性/负向类似表达：
        正向：整体是满意,愉快,赞扬,积极的
        中性：没有明显情绪,或者只是描述事实
        负向：整体是不满，失望，愤怒，讽刺，抱怨
        请按照以下格式输出:
1.分类: 正向 | 理由: ...
2.分类: 负向 | 理由: ...
3.分类: 中性 | 理由: ...
4.分类: 中性 | 理由: ...
5.分类: 负向 | 理由: ...

        文本如下：
{pharse}
        """
        response = request_to_model(query_str, is_classify=False)

        # 解析模型返回结果
        lines = response.strip().split("\n")
        batch_results = []

        # 模型输出可能会输出空行,多行（例如理由行也作为一行)要进行清洗逻辑
        valid_lines = [line for line in lines if "分类" in line]
        valid_lines = valid_lines[-5:]

        for i, line in enumerate(valid_lines):
            try:
                label_part = line.split("分类:")[1].split("|")[0].strip()
            except IndexError:
                label_part = "未知"

            batch_results.append({
                'text': batch_df.iloc[i]['text'],
                'label': label_map[label_part]
            })
        print("##########")
        print(response)
        # print(batch_df)
        # 直接合并当前批次
        df_ai = pd.concat([df_ai, pd.DataFrame(batch_results)], ignore_index=True)

        # 去重避免重复保存
        df_ai = df_ai.drop_duplicates(subset=['text'], keep='last')

        # 更新进度(需要时刻更新已经处理完的索引,还需要正确更新processed)
        processed = processed + batch_df.index.tolist()
        save_processed_indices(processed)
        print("进度已更新")

        # 保存进度
        df_ai.to_csv(OUTPUT_FILE, index=False)
        print(f"💾 已累计保存 {len(df_ai)} 条标注数据到 {OUTPUT_FILE}")
        # print(df_annotated)
        # break


# 豆包API
def request_doubao(self, content):
    from openai import OpenAI

    # openai_api_key = "d1b18464-b9cf-4b7e-9a19-6a17b37ee74c"
    openai_api_key = ""  # TODO: replace your api key
    client = OpenAI(
        api_key=openai_api_key,
        base_url="",  # TODO: replace your api base_url
    )

    # Non-streaming:
    messages = [{"role": "user", "content": "{}".format(content)}]
    completion = client.chat.completions.create(
        # model="ep-20250212103518-lhkk7",  # your model endpoint ID
        # model="ep-20250402144717-grkgt",  # your model endpoint ID
        model="ep-20250402113148-cdzlc",  # your model endpoint ID
        messages=messages,
        stream=False,  # 关键参数启用流式
    )
    res_content = completion.choices[0].message.content

    # print("content:", res_content)
    return res_content


if __name__ == "__main__":
    if RESET:
        reset_progress()
    else:
        glm_emotion_annotate(CSV_FILE)

    print("end")