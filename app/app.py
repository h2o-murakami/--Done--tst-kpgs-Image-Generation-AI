import os
import logging
  
import gradio as gr
import google.cloud.logging
import vertexai
from vertexai.language_models import ChatModel, ChatMessage, InputOutputTextPair
  
   
# Cloud Logging �n���h���� logger �ɐڑ�
logger = logging.getLogger()
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
  
# �萔�̒�`
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
  
# vertexai �C���X�^���X�̏�����
vertexai.init(project=PROJECT_ID, location="us-central1")
  
  
def llm_chat(message, chat_history):
    # ��Ճ��f����ݒ�
    chat_model = ChatModel.from_pretrained("chat-bison@001")
  
    # �p�����[�^���w��
    parameters = {
        "max_output_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40
    }
  
    # ��b�����̃��X�g��������
    message_history = []
  
    # ��b�����̃t�H�[�}�b�g�𐮌`
    for row in chat_history:
        input_from_user = row[0]
        output_from_llm = row[1]
  
        q_message = ChatMessage(author="user", content=input_from_user)
        message_history.append(q_message)
  
        a_message = ChatMessage(author="llm", content=output_from_llm)
        message_history.append(a_message)
  
    # ��Ճ��f���ɉ�b�������C���v�b�g
    chat = chat_model.start_chat(message_history=message_history)
  
    # ��Ճ��f���Ƀv�����v�g���N�G�X�g�𑗐M
    response = chat.send_message(message, **parameters)
  
    return response.text
  
  
def respond(message, chat_history):
    # llm �ŉ񓚂𐶐�
    bot_message = llm_chat(message, chat_history)
  
    # Cloud Logging �������ݗp
    logger.info(f"message: {message}")
    logger.info(f"chat_history: {chat_history}")
    logger.info(f"bot_message: {bot_message}")
  
    chat_history.append((message, bot_message))
    return "", chat_history
  
# gradio �̐ݒ�
with gr.Blocks() as llm_web_app:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
  
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
  
# Gradio �̗����グ
llm_web_app.launch(server_name="0.0.0.0", server_port=7860)