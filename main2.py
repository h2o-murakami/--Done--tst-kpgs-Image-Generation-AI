import io
import traceback
import gradio as gr
import vertexai
from vertexai.preview.vision_models import Image,ImageGenerationModel,ImageGenerationResponse
  
# 環境変数の設定
PROJECT_ID = "tst-kpgs-poc-h2o"  # Google Cloud プロジェクトの ID
LOCATION = "us-central1"  # Gemini モデルを使用するリージョン
  
vertexai.init(project=PROJECT_ID, location=LOCATION)
  
# 画像の前処理
def get_bytes_from_pil(image: Image) -> bytes:
    byte_io_png = io.BytesIO()
    image.save(byte_io_png, "PNG")
    return byte_io_png.getvalue()
  
# 画像の生成処理
def imagen_generate(
    base_image,
    mask_mode: str,
    edit_mode: str,
    prompt: str,
    negative_prompt: str,
):
    # 使用するモデルの指定
    IMAGE_GENERATION_MODEL = "imagegeneration@006"
    generation_model = ImageGenerationModel.from_pretrained(IMAGE_GENERATION_MODEL)
  
    image_pil = Image(image_bytes=get_bytes_from_pil(base_image))
  
    generate_response: ImageGenerationResponse = generation_model.edit_image(
        prompt=prompt,
        base_image=image_pil,
        negative_prompt=negative_prompt,
        number_of_images=4,
        edit_mode=edit_mode,
        mask_mode=mask_mode,
        language="ja",  # 日本語でのプロンプトに対応するために追加
    )
  
    return [img._pil_image for img in generate_response.images]
  
# Update function called by Gradio
def update(
    base_image,
    mask_mode,
    edit_mode,
    prompt,
    negative_prompt,
):
  
    if len(negative_prompt) == 0:
        negative_prompt = None
  
    print("prompt:", prompt)
    print("negative_prompt:", negative_prompt)
  
    images = []
    error_message = ""
  
    try:
        images = imagen_generate(base_image, mask_mode, edit_mode, prompt, negative_prompt)
    except Exception as e:
        print(e)
        error_message = """
        An error occured calling the API.
        1. Check if response was not blocked based on policy violation, check if the UI behaves the same way.
        2. Try a different prompt to see if that was the problem.
        """
        error_message += "\n" + traceback.format_exc()
  
    return images, error_message
  
  
# gradio の設定
iface = gr.Interface(
    # 使用する関数
    fn=update,
    # 入力タイプ：テキストと画像
    inputs=[
        gr.Image(
            label="アップロードした画像",
            type="pil",
        ),
        gr.Dropdown(
            label="マスクモード",
            choices=["foreground", "background"],
            value="background",
        ),
        gr.Dropdown(
            label="編集モード",
            choices=["product-image", "inpainting-insert","inpainting-remove","outpainting"],
            value="product-image",
        ),
        gr.Textbox(
            label="プロンプト入力",  # 日本語での表示に修正
            # 日本語での説明文章に修正
            placeholder="短い文とキーワードをカンマで区切って使用する",
            value="",
        ),
        gr.Textbox(
            label="ネガティブプロンプト",  # 日本語での表示に修正
            # 日本語での説明文章に修正
            placeholder="表示したくない内容を定義する",
            value="",
        ),
    ],
    # 出力タイプ：画像
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
    # 日本語での説明文章に修正
    title="Image modify with Imagen on Vertex AI",  # タイトルの修正
    description="""画像から背景（前景）を認識し、プロンプトの内容に沿った画像を生成します。Imagen のドキュメントについては、この[リンク](https://cloud.google.com/vertex-ai/docs/generative-ai/image/generate-images)を参照してください。 """,
    allow_flagging="never",
    theme=gr.themes.Soft(),
)
  
# # Local 起動
# iface.launch()
  
# Cloud Run 起動
iface.launch(server_name="0.0.0.0", server_port=10080)