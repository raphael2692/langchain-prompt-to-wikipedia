
# standard
import os
import json
import re

from loguru import logger
import trafilatura

from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from typing import Dict, List
# from langchain.chains import SequentialChain
# from langchain.memory import SimpleMemory

from templates import URL_TEMPLATE, SUMMARY_TEMPLATE

class TrafilaturaChain(Chain):
    input_key: str = "url"  # :meta private:
    output_key: str = "text" # :meta private:

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _get_text(self, url):
        downloaded = trafilatura.fetch_url(url)
        result = json.loads(trafilatura.extract(
            downloaded, output_format='json', include_links=False))
        text = result["text"].replace("\n", "")
        paragraph = self._get_paragraphs_from_text(text, 500)
        return "".join(paragraph[0:10])

    def _get_paragraphs_from_text(self, text, max_length):
        sentences = re.findall('[^\.!?]+[\.!?]', text)
        current_paragraph = ""
        current_length = 0
        paragraphs = []
        for sentence in sentences:
            if current_length + len(sentence) <= max_length:
                current_paragraph += sentence.strip() + " "
                current_length += len(sentence)
            else:
                paragraphs.append(current_paragraph.strip())
                current_paragraph = sentence.strip() + " "
                current_length = len(sentence)
        paragraphs.append(current_paragraph.strip())
        return paragraphs

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        return {self.output_key: self._get_text(inputs['url'])}
    
def init_llm_chain(prompt_template, template_variables, chat_temperature, output_key):
    human_messsage_prompt_template = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=prompt_template,
            input_variables=[var for var in template_variables],
        )
    )
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [human_messsage_prompt_template])
    chat = ChatOpenAI(temperature=chat_temperature)
    return LLMChain(llm=chat, prompt=chat_prompt_template, output_key=output_key, verbose=True)

if __name__ == "__main__":
    logger.info("Starting app...")
    # read settings
    with open("secrets.json", "r") as jsonfile:
        secrets = json.load(jsonfile)

    # set openai key
    os.environ["OPENAI_API_KEY"] = secrets["openai_api_key"]
    
    # init chat object
    chat = ChatOpenAI(temperature=0)

    # setup url chain
    url_chain_human_prompt_template = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template = URL_TEMPLATE['text'],
            input_variables = URL_TEMPLATE['input_variables'],
        )
    )
    url_chain_chat_prompt_template = ChatPromptTemplate.from_messages(
        [url_chain_human_prompt_template])

    url_chain = LLMChain(llm=chat, prompt=url_chain_chat_prompt_template, output_key="url", verbose=True)

    # setup trafilatura chain
    trafilatura_chain = TrafilaturaChain()

    # setup summary chain
    summary_chain_human_prompt_template = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template = SUMMARY_TEMPLATE['text'],
            input_variables = SUMMARY_TEMPLATE['input_variables'],
        )
    )
    summary_chain_chat_prompt_template = ChatPromptTemplate.from_messages(
        [summary_chain_human_prompt_template])

    logger.info("Composing chain...")
    summary_chain = LLMChain(llm=chat, prompt=summary_chain_chat_prompt_template, output_key="summary", verbose=True) 

    simple_sequential_chain = SimpleSequentialChain(
        chains=[url_chain, trafilatura_chain, summary_chain],
        verbose=True
    )
    logger.info("Fetching user input...")
    user_prompt = input("\n What do you want to search in Wikipedia?: ")
    simple_sequential_chain.run(user_prompt)


