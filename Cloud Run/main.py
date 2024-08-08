import gradio as gr
import traceback
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
  
# ���ϐ��̐ݒ�
PROJECT_ID = "tst-kpgs-poc-h2o"  # Google Cloud �v���W�F�N�g�� ID
LOCATION = "us-central1"  # Gemini ���f�����g�p���郊�[�W����
  
vertexai.init(project=PROJECT_ID, location=LOCATION)
  
  
def imagen_generate(
    model_name: str,
    prompt: str,
    negative_prompt: str,
    sampleImageSize: int,
    aspect_ratio: str, # �A�X�y�N�g����w��ł���悤�ɒǉ�
    sampleCount: int,
    seed=None,
):
    model = ImageGenerationModel.from_pretrained(model_name)
    generate_response = model.generate_images(
        prompt=prompt,
        negative_prompt=negative_prompt,
        number_of_images=sampleCount,
        guidance_scale=float(sampleImageSize),
        aspect_ratio=aspect_ratio, # �A�X�y�N�g����w��ł���悤�ɒǉ�
        language="ja", # ���{��ł̃v�����v�g�ɑΉ����邽�߂ɒǉ�
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
    aspect_ratio="1:1", # �A�X�y�N�g����w��ł���悤�ɒǉ�
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
            model_name, prompt, negative_prompt, sampleImageSize, aspect_ratio, sampleCount, seed # �A�X�y�N�g����w��ł���悤�ɒǉ�
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
  
# gradio �̐ݒ�
iface = gr.Interface(
    fn=update,
    inputs=[
        gr.Dropdown(
            label="�g�p���郂�f��",
            choices=["imagegeneration@002", "imagegeneration@006"], # �ŐV���f�����g�p����p�ɏC��
            value="imagegeneration@006", # �ŐV���f�����g�p����p�ɏC��
            ),
        gr.Textbox(
            label="�v�����v�g����", # ���{��ł̕\���ɏC��
            # ���{��ł̐������͂ɏC��
            placeholder="�Z�����ƃL�[���[�h���J���}�ŋ�؂��Ďg�p����B���Ƃ��΁u����, ��󂩂�̃V���b�g, �����Ă��钹�v�Ȃ�",
            value="",
            ),
        gr.Textbox(
            label="�l�K�e�B�u�v�����v�g", # ���{��ł̕\���ɏC��
            # ���{��ł̐������͂ɏC��
            placeholder="�\���������Ȃ����e���`���܂�",  
            value="",
            ),
        gr.Dropdown(
            label="�o�̓C���[�W�T�C�Y", # ���{��ł̕\���ɏC��
            choices=["256", "1024", "1536"],
            value="1536",
            ),
        gr.Dropdown(
            # �A�X�y�N�g����w��ł���悤�ɒǉ�
            label="�A�X�y�N�g��", # ���{��ł̕\���ɏC��
            choices=["1:1", "9:16", "16:9","3:4", "4:3"],
            value="1:1",
            ),
        gr.Number(
            label="�\������",  # ���{��ł̕\���ɏC��
            # ���{��ł̐������͂ɏC��
            info="���������摜�̐��B�w��ł��鐮���l: 1�`4�B�f�t�H���g�l: 4",
            value=4),
        gr.Number(
            label="seed",
            # ���{��ł̐������͂ɏC��
            info="�K�v�ɉ����Č��ʂ��Č��ł���悤�ɁA�\�ł���΃V�[�h���g�p���Ă��������B�����͈�: (0, 2147483647)",
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
    title="Image Generation with Imagen on Vertex AI", # �^�C�g���̏C��
    # ���{��ł̐������͂ɏC�� 
    description="""�e�L�X�g�v�����v�g����̉摜�����BImagen �̃h�L�������g�ɂ��ẮA����[�����N](https://cloud.google.com/vertex-ai/docs/generative-ai/image/generate-images)���Q�Ƃ��Ă��������B """,
    allow_flagging="never",
    theme=gr.themes.Soft(),
)
  
# Local �N��
iface.launch()