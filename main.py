# 必要なライブラリのインポート

import gradio as gr

import traceback

import vertexai

from vertexai.preview.vision_models import ImageGenerationModel

  

# 環境変数の設定

PROJECT_ID = "tst-kpgs-poc-h2o"  # Google Cloud プロジェクトの ID

LOCATION = "asia-northeast1"  # Gemini モデルを使用するリージョン

  

# Vertex AI の初期化

vertexai.init(project=PROJECT_ID, location=LOCATION)

  

  

# 入力されたプロンプトに基づいて画像を生成する関数

def imagen_generate(

    model_name: str,

    prompt: str,

    negative_prompt: str,

    sampleImageSize: int,

    aspect_ratio: str, # アスペクト比を指定できるように追加

    sampleCount: int,

    seed=None,

):

    # 指定された名前の学習済みモデルを読み込む

    model = ImageGenerationModel.from_pretrained(model_name)

    # 読み込んだモデルを使って画像を生成

    generate_response = model.generate_images(

        prompt=prompt,

        negative_prompt=negative_prompt,

        number_of_images=sampleCount,

        guidance_scale=float(sampleImageSize),

        aspect_ratio=aspect_ratio, # アスペクト比を指定できるように追加

        language="ja", # 日本語でのプロンプトに対応するために追加

        seed=seed,

    )

    # 生成された画像を格納するためのリストを作成

    images = []

    # 生成された画像を順番に処理

    for index, result in enumerate(generate_response):

        # 生成された画像をリストに追加

        images.append(generate_response[index]._pil_image)

    # 生成された画像のリストと、生成処理のレスポンスを返却

    return images, generate_response

  

  

# Gradio のインターフェースが更新された際に呼び出される関数

# 引数は、Gradio のインターフェースから入力されたデータ

def update(

    model_name,

    prompt,

    negative_prompt,

    sampleImageSize="1536",

    aspect_ratio="1:1", # アスペクト比を指定できるように追加

    sampleCount=4,

    seed=None,

):

    # ネガティブプロンプトが入力されていない場合は、`None` を設定

    if len(negative_prompt) == 0:

        negative_prompt = None

  

    print("prompt:", prompt)

    print("negative_prompt:", negative_prompt)

  

    # シード値に無効な値が入力された場合は、`None` を設定

    if seed < 0 or seed > 2147483647:

        seed = None

  

    # 生成された画像を受け取るためのリストを作成

    images = []

    # エラーメッセージを受け取るための変数を定義

    error_message = ""

    try:

        # imagen_generate関数を呼び出して画像を生成

        images, generate_response = imagen_generate(

            model_name, prompt, negative_prompt, sampleImageSize, aspect_ratio, sampleCount, seed # アスペクト比を指定できるように追加

        )

    # 画像生成処理に失敗した場合の例外処理

    except Exception as e:

        print(e)

        # エラーメッセージを設定

        error_message = """An error occured calling the API.

      1. Check if response was not blocked based on policy violation, check if the UI behaves the same way.

      2. Try a different prompt to see if that was the problem.

      """

        error_message += "\n" + traceback.format_exc()



    # 生成された画像とエラーメッセージを返却  

    return images, error_message

  

# Gradio のインターフェース設定

iface = gr.Interface(

    # インターフェースが更新された際に呼び出される関数を指定

    fn=update,

    # インターフェースへの入力要素を指定

    inputs=[

        gr.Dropdown(

            label="使用するモデル",

            choices=["imagegeneration@002", "imagegeneration@006"], # 最新モデルを使用する用に修正

            value="imagegeneration@006", # 最新モデルを使用する用に修正

            ),

        gr.Textbox(

            label="プロンプト入力", # 日本語での表示に修正

            # 日本語での説明文章に修正

            placeholder="短い文とキーワードをカンマで区切って使用する。たとえば「昼間, 上空からのショット, 動いている鳥」など",

            value="",

            ),

        gr.Textbox(

            label="ネガティブプロンプト", # 日本語での表示に修正

            # 日本語での説明文章に修正

            placeholder="表示したくない内容を定義します",  

            value="",

            ),

        gr.Dropdown(

            label="出力イメージサイズ", # 日本語での表示に修正

            choices=["256", "1024", "1536"],

            value="1536",

            ),

        gr.Dropdown(

            # アスペクト比を指定できるように追加

            label="アスペクト比", # 日本語での表示に修正

            choices=["1:1", "9:16", "16:9","3:4", "4:3"],

            value="1:1",

            ),

        gr.Number(

            label="表示件数",  # 日本語での表示に修正

            # 日本語での説明文章に修正

            info="生成される画像の数。指定できる整数値: 1～4。デフォルト値: 4",

            value=4),

        gr.Number(

            label="シード",

            # 日本語での説明文章に修正

            info="必要に応じて結果を再現できるように、可能であればシードを使用してください。整数範囲: (0, 2147483647)",

            value=-1,

        ),

    ],

    # インターフェースの出力要素を指定

    outputs=[

        gr.Gallery(

            label="生成された画像",

            show_label=True,

            elem_id="gallery",

            columns=[2],

            object_fit="contain",

            height="auto",

        ),

        gr.Textbox(label="エラーメッセージ"),

    ],

    # インターフェースのタイトルを設定

    title="キンプリ＆胡麻鯖 AI Canvas",

    # インターフェースの説明文を設定

    description="""テキストプロンプトからの画像生成。Imagen のドキュメントについては、この[リンク](https://cloud.google.com/vertex-ai/docs/generative-ai/image/generate-images)を参照してください。 """,

    # フラグ機能(ユーザーのフィードバック送信の許可設定)を無効に設定

    allow_flagging="never",

    # インターフェースのテーマを設定

    theme=gr.themes.Soft(),

)

  

# Local 起動

iface.launch()