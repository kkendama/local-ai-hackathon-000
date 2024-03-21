# パッケージのインポート
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from tqdm import tqdm
import pandas as pd

model_name = "tokyotech-llm/Swallow-MX-8x7b-NVE-v0.1"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    add_eos_token=True,
    use_fast=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=False,
)

example1_user = """次の文章を読んで、AIへの指示文を考えてください。
総合情報詳細経路情報詳細風速情報アクセス解析マップ全画像一覧 || ニュース災害情報ブログトラックバック台風前線ツイフーン台風なう！台風空想リソースGoogle EarthGE AnimationAtomフィードGeoJSON"""
example1_assistant = """1週間後に台風が首都圏を直撃するようなのですが、その日はどうしても外せない予定があり、外出をすることになっています。
安全に外出を行うため、それまでに買っておいた方が良いものがあれば教えていただけますか？
また、注意しておくべきこともまとめてください。"""

example2_user = """次の文章を読んで、AIへの指示文を考えてください。
※１ 上記数値はお引渡し時のフィルター初期性能を示しています。（東レ㈱調べ）PM2.5は粒子径2.5μｍ以下、PM0.5は粒子径0.5μm以下の大気中浮遊微粒子を指しますが、試験環境上、0.5μm以上の微粒子にて測定を行っております。よって、PM2.5及びPM0.5すべてに有効というわけではありません。★家の隙間、衣服への付着、窓・玄関の開閉によって室内へ侵入する花粉・微粒子等は対象となりません。また、花粉症・呼吸器疾患などの治療や改善を目的とするものではありません。
●当社モデルプラン(約140㎡、UA値=0.67W/m²K)における温熱環境シミュレーション結果 ●外気条件:東京1月の代表的な1日 ●生活スケジュール:建築学会標準生活スケジュールに準拠 ●暖房スケジュール
■1階快適エアリー;6:00~10:00、12:00~14:00、16:00~22:00 暖房運転(21℃)その他時間はキープ運転(17℃)■2階快適エアリー;20:00~23:00 暖房運転(21℃)その他時間はキープ運転(17℃)"""
example2_assistant = """寒さが厳しくなってきました。暖房費を抑えるために、省エネで効率的な暖房方法を知りたいです。"""

example3_user = """次の文章を読んで、AIへの指示文を考えてください。
ゴルフの専門店などに行くといろんな種類の物を見つける事ができます。多くの場合にはスタンド型とカート型があります。スタンド型はその名の通り、立てかける事ができます。軽くて小さいので気軽にゴルフを楽しみたいという人にとっては向いているでしょう。"""
example3_assistant = """ゴルフクラブを購入しようと思っているのですが、スタンド型とカート型のどちらを選ぶべきでしょうか？
それぞれのメリットとデメリットについて教えてください。
また、初心者はどちらを選ぶべきか、詳しい方の意見を伺うことは可能でしょうか？"""

def generate_prompt(text):
    prompt= f'''あなたはイラストレーターです。自分のホームページを作りたいと思っています。どのような内容を掲載していくべきか、そして、どのようなデザインにすべきか意見をください。 また、自分のイラストを載せるだけでなく、他のクリエイターとの交流も深めたいと思っています。SNSを活用したコミュニティや、オンラインでのアートコミュニティについてもアドバイスをもらいたいです。 さらに、自分の技術向上のため、他のアーティストとの交流や、自分自身のアイデアを広げるためのアプローチ方法も提案していただきたいです。'''
    return prompt

text = """腰痛解消・熟睡・快眠マットレスの選び方からベッドメーカーの徹底比較と評判・口コミまで情報満載。テンピュール、シモンズマットレス、フランスベッド、マニフレックス、シーリー、日本ベッド、トゥルースリーパー、センベラマットレスの評判・口コミ情報満載の快眠情報サイトです。
また横向きで寝る方にもおすすめしていました。肩や肘がしびれたり、うっ血せず長時間でも横の姿勢を維持できるのです。 イトーヨーカドーで調度体感しているご年配の方がいらしたのですが、すごいすごいと驚いていました。"""

def chat(prompt):
    # メッセージリストの準備
    messages = [
        {"role": "user", "content": example1_user},
        {"role": "assistant", "content": example1_assistant},
        {"role": "user", "content": example2_user},
        {"role": "assistant", "content": example2_assistant},
        {"role": "user", "content": example3_user},
        {"role": "assistant", "content": example3_assistant},
        {"role": "user", "content": generate_prompt(text)},
    ]

    # 推論の実行
    with torch.no_grad():
      token_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
      output_ids = model.generate(
          token_ids.to(model.device),
          temperature=0.7,
          do_sample=True,
          #top_p=0.95,
          #top_k=40,
          max_new_tokens=128,
          repetition_penalty=1.05
      )
    output = tokenizer.decode(output_ids[0][token_ids.size(1) :])
    return output

prompt = generate_prompt("")
print(chat(prompt))
