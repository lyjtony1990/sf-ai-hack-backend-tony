import os
import time
from dataclasses import dataclass

import openai
import pinecone
import yaml
from llama_index import GPTVectorStoreIndex, Response
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate
from llama_index.vector_stores import PineconeVectorStore


@dataclass
class IndexConfig:
    name: str
    pinecone_api_key: str
    page_ids: []


def initialize_index(index_config):
    pinecone.init(api_key=index_config.pinecone_api_key, environment="gcp-starter")
    pinecone_index = pinecone.Index(index_config.name)
    vector_store = PineconeVectorStore(pinecone_index)
    return GPTVectorStoreIndex.from_vector_store(vector_store=vector_store)


class IndexManager:
    def __init__(self):
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        os.environ['OPENAI_API_KEY'] = config['openai_api_key']
        os.environ['NOTION_INTEGRATION_TOKEN'] = config['notion_integration_token']

        indexes = config.get('indexes', [])
        self.indexMap = {}

        for index in indexes:
            index_config: IndexConfig = IndexConfig(index.get('name', ''),
                                                    index.get('pinecone_api_key', ''),
                                                    index.get('page_ids', []))
            self.indexMap[index.get('name', '')] = initialize_index(index_config)

    def get_response(self, index, question):
        template = (
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Given this information, please answer the question: {query_str}.\n"
            "If the context provides insufficient information and the question cannot be directly answered, "
            "reply 'I am lacking knowledge to answer this questions. "
            "Please enter manually then add it to knowledge base.'"
        )

        qa_template = PromptTemplate(template)
        index = self.indexMap[index]
        if index is None:
            return f"index '{index} not found!'"

        query_engine = index.as_query_engine(text_qa_template=qa_template,
                                             response_mode='compact', llm=OpenAI(model="gpt-3.5-turbo"),
                                             similarity_top_k=2)

        start_time = time.time()
        res: Response = query_engine.query(question)
        print(f"time taken: {(time.time() - start_time)}seconds")
        print(res)
        return res.response

    def process(self, index, user_query):
        openAI_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant, skilled in extracting information from an email. "
                            "Remove the html and extract the text from this."},
                {"role": "user", "content": user_query}
            ]

        )
        print(openAI_response)

        email_question = openAI_response.choices[0].message["content"]
        template = (
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Given this information, please answer the question: {query_str}.\n"
            "If the context provides insufficient information and the question cannot be directly answered, "
            "reply 'I am lacking knowledge to answer this questions. "
            "Please enter manually then add it to knowledge base.'"
        )

        qa_template = PromptTemplate(template)
        if index is None:
            return f"index '{index} not found!'"

        query_engine = index.as_query_engine(text_qa_template=qa_template,
                                             response_mode='refine', llm=OpenAI(model="gpt-3.5-turbo"))

        start_time = time.time()
        res: Response = query_engine.query(email_question)
        print(f"time taken: {(time.time() - start_time)}seconds")
        print(res)
        return res.response
