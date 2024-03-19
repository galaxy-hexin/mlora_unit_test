import json
import logging
import os.path as osp
from typing import Union

'''
这个类比较简单，主要是对训练的样本数据进行提示词工程
_init_：使用文件地址打开不同的模板文件
generate_prompt：根据是否有输入，使用不同的模板把input替换成不同的字符
get_response:输出模板输出就行
'''


# manage templates and prompt building.
class Prompter:
    def __init__(self, file_name: str):
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        logging.info(
            f"Using prompt template {file_name}: {self.template['description']}")

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}\n"
        logging.debug(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[-1].strip()
