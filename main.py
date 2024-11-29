# �K�v�ȃ��C�u�����̃C���|�[�g

import gradio as gr

import traceback

import vertexai

from vertexai.preview.vision_models import ImageGenerationModel

  

# ���ϐ��̐ݒ�

PROJECT_ID = "tst-kpgs-poc-h2o"  # Google Cloud �v���W�F�N�g�� ID

LOCATION = "asia-northeast1"  # Gemini ���f�����g�p���郊�[�W����

  

# Vertex AI �̏�����

vertexai.init(project=PROJECT_ID, location=LOCATION)

  

  

# ���͂��ꂽ�v�����v�g�Ɋ�Â��ĉ摜�𐶐�����֐�

def imagen_generate(

    model_name: str,

    prompt: str,

    negative_prompt: str,

    sampleImageSize: int,

    aspect_ratio: str, # �A�X�y�N�g����w��ł���悤�ɒǉ�

    sampleCount: int,

    seed=None,

):

    # �w�肳�ꂽ���O�̊w�K�ς݃��f����ǂݍ���

    model = ImageGenerationModel.from_pretrained(model_name)

    # �ǂݍ��񂾃��f�����g���ĉ摜�𐶐�

    generate_response = model.generate_images(

        prompt=prompt,

        negative_prompt=negative_prompt,

        number_of_images=sampleCount,

        guidance_scale=float(sampleImageSize),

        aspect_ratio=aspect_ratio, # �A�X�y�N�g����w��ł���悤�ɒǉ�

        language="ja", # ���{��ł̃v�����v�g�ɑΉ����邽�߂ɒǉ�

        seed=seed,

    )

    # �������ꂽ�摜���i�[���邽�߂̃��X�g���쐬

    images = []

    # �������ꂽ�摜�����Ԃɏ���

    for index, result in enumerate(generate_response):

        # �������ꂽ�摜�����X�g�ɒǉ�

        images.append(generate_response[index]._pil_image)

    # �������ꂽ�摜�̃��X�g�ƁA���������̃��X�|���X��ԋp

    return images, generate_response

  

  

# Gradio �̃C���^�[�t�F�[�X���X�V���ꂽ�ۂɌĂяo�����֐�

# �����́AGradio �̃C���^�[�t�F�[�X������͂��ꂽ�f�[�^

def update(

    model_name,

    prompt,

    negative_prompt,

    sampleImageSize="1536",

    aspect_ratio="1:1", # �A�X�y�N�g����w��ł���悤�ɒǉ�

    sampleCount=4,

    seed=None,

):

    # �l�K�e�B�u�v�����v�g�����͂���Ă��Ȃ��ꍇ�́A`None` ��ݒ�

    if len(negative_prompt) == 0:

        negative_prompt = None

  

    print("prompt:", prompt)

    print("negative_prompt:", negative_prompt)

  

    # �V�[�h�l�ɖ����Ȓl�����͂��ꂽ�ꍇ�́A`None` ��ݒ�

    if seed < 0 or seed > 2147483647:

        seed = None

  

    # �������ꂽ�摜���󂯎�邽�߂̃��X�g���쐬

    images = []

    # �G���[���b�Z�[�W���󂯎�邽�߂̕ϐ����`

    error_message = ""

    try:

        # imagen_generate�֐����Ăяo���ĉ摜�𐶐�

        images, generate_response = imagen_generate(

            model_name, prompt, negative_prompt, sampleImageSize, aspect_ratio, sampleCount, seed # �A�X�y�N�g����w��ł���悤�ɒǉ�

        )

    # �摜���������Ɏ��s�����ꍇ�̗�O����

    except Exception as e:

        print(e)

        # �G���[���b�Z�[�W��ݒ�

        error_message = """An error occured calling the API.

      1. Check if response was not blocked based on policy violation, check if the UI behaves the same way.

      2. Try a different prompt to see if that was the problem.

      """

        error_message += "\n" + traceback.format_exc()



    # �������ꂽ�摜�ƃG���[���b�Z�[�W��ԋp  

    return images, error_message

  

# Gradio �̃C���^�[�t�F�[�X�ݒ�

iface = gr.Interface(

    # �C���^�[�t�F�[�X���X�V���ꂽ�ۂɌĂяo�����֐����w��

    fn=update,

    # �C���^�[�t�F�[�X�ւ̓��͗v�f���w��

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

            label="�V�[�h",

            # ���{��ł̐������͂ɏC��

            info="�K�v�ɉ����Č��ʂ��Č��ł���悤�ɁA�\�ł���΃V�[�h���g�p���Ă��������B�����͈�: (0, 2147483647)",

            value=-1,

        ),

    ],

    # �C���^�[�t�F�[�X�̏o�͗v�f���w��

    outputs=[

        gr.Gallery(

            label="�������ꂽ�摜",

            show_label=True,

            elem_id="gallery",

            columns=[2],

            object_fit="contain",

            height="auto",

        ),

        gr.Textbox(label="�G���[���b�Z�[�W"),

    ],

    # �C���^�[�t�F�[�X�̃^�C�g����ݒ�

    title="�L���v�����Ӗ��I AI Canvas",

    # �C���^�[�t�F�[�X�̐�������ݒ�

    description="""�e�L�X�g�v�����v�g����̉摜�����BImagen �̃h�L�������g�ɂ��ẮA����[�����N](https://cloud.google.com/vertex-ai/docs/generative-ai/image/generate-images)���Q�Ƃ��Ă��������B """,

    # �t���O�@�\(���[�U�[�̃t�B�[�h�o�b�N���M�̋��ݒ�)�𖳌��ɐݒ�

    allow_flagging="never",

    # �C���^�[�t�F�[�X�̃e�[�}��ݒ�

    theme=gr.themes.Soft(),

)

  

# Local �N��

iface.launch()