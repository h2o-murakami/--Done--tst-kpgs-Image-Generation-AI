import gradio as gr
import traceback
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
  
# 環境変数の設定
PROJECT_ID = "tst-kpgs-poc-h2o"  # Google Cloud プロジェクトの ID
LOCATION = "us-central1"  # Gemini モデルを使用するリージョン
  
vertexai.init(project=PROJECT_ID, location=LOCATION)
  
  
def imagen_generate(
    model_name: str,
    prompt: str,
    negative_prompt: str,
    sampleImageSize: int,
    aspect_ratio: str, # アスペクト比を指定できるように追加
    sampleCount: int,
    seed=None,
):
    model = ImageGenerationModel.from_pretrained(model_name)
    generate_response = model.generate_images(
        prompt=prompt,
        negative_prompt=negative_prompt,
        number_of_images=sampleCount,
        guidance_scale=float(sampleImageSize),
        aspect_ratio=aspect_ratio, # アスペクト比を指定できるように追加
        language="ja", # 日本語でのプロンプトに対応するために追加
        seed=seed,
    )
    images = []
    for index, result in enumerate(generate_response):
        images.append(generate_response[index]._pil_image)
    return images, generate_response
  
  
# Update function called by Gradio
def update(
    model_name,
    prompt,
    negative_prompt,
    sampleImageSize="1536",
    aspect_ratio="1:1", # アスペクト比を指定できるように追加
    sampleCount=4,
    seed=None,
):
    if len(negative_prompt) == 0:
        negative_prompt = None
  
    print("prompt:", prompt)
    print("negative_prompt:", negative_prompt)
  
    # Advanced option, try different the seed numbers
    # any random integer number range: (0, 2147483647)
    if seed < 0 or seed > 2147483647:
        seed = None
  
    # Use & provide a seed, if possible, so that we can reproduce the results when needed.
    images = []
    error_message = ""
    try:
        images, generate_response = imagen_generate(
            model_name, prompt, negative_prompt, sampleImageSize, aspect_ratio, sampleCount, seed # アスペクト比を指定できるように追加
        )
    except Exception as e:
        print(e)
        error_message = """An error occured calling the API.
      1. Check if response was not blocked based on policy violation, check if the UI behaves the same way.
      2. Try a different prompt to see if that was the problem.
      """
        error_message += "\n" + traceback.format_exc()
        # raise gr.Error(str(e))
  
    return images, error_message
  
# gradio の設定
iface = gr.Interface(
    fn=update,
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
            label="seed",
            # 日本語での説明文章に修正
            info="必要に応じて結果を再現できるように、可能であればシードを使用してください。整数範囲: (0, 2147483647)",
            value=-1,
        ),
    ],
    outputs=[
        gr.Gallery(
            label="Generated Images",
            show_label=True,
            elem_id="gallery",
            columns=[2],
            object_fit="contain",
            height="auto",
        ),
        gr.Textbox(label="Error Messages"),
    ],
    title="Image Generation with Imagen on Vertex AI", # タイトルの修正
    # 日本語での説明文章に修正 
    description="""テキストプロンプトからの画像生成。Imagen のドキュメントについては、この[リンク](https://cloud.google.com/vertex-ai/docs/generative-ai/image/generate-images)を参照してください。 """,
    allow_flagging="never",
    theme=gr.themes.Soft(),
)
  
# Local 起動
iface.launch()