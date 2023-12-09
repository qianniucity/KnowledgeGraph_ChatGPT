import prompt
import util
import config
import types_enum

class Service:
    def __init__(self):
        super(Service, self).__init__()
        self.util = util.Util()
        self.configs = config.ConfigParser()


    def extract_information(self, message, history):
        # 1.组装系统提示，历史对话，用户当前问题
        system_prompt = prompt.SIMPLE_SYSTEM_PROMPT
        
        # 2.组装图谱提示
        graph_prompt = prompt.SIMPLE_GRAPH_PROMPT.format(
            entity_types=types_enum.entity_types,
            relation_types=types_enum.relation_types,
            specification=message
        )

        messages = self.util.concat_chat_message(system_prompt, history, graph_prompt)

        # 2. 去调用 OpenAI 的接口完成任务
        response = self.util.ChatOpenAI(messages)

        return response.content