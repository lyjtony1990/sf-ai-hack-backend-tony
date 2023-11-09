import os
import time

import openai
import pinecone
import yaml
from llama_index.llms import OpenAI
from llama_index.vector_stores import PineconeVectorStore
from llama_index import GPTVectorStoreIndex, StorageContext, ServiceContext, download_loader, Response
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.prompts import PromptTemplate


class IndexManager:
    def __init__(self):
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        os.environ['OPENAI_API_KEY'] = config['openai_api_key']
        os.environ['NOTION_INTEGRATION_TOKEN'] = config['notion_integration_token']

        pinecone.init(api_key=config['pinecone_api_key'], environment="gcp-starter")

        self.index_name = config['index_name']
        self.initialize_index()

    def initialize_index(self):
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(self.index_name, dimension=1536, metric='cosine')

        pinecone_index = pinecone.Index(self.index_name)
        self.vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.embed_model = OpenAIEmbedding(model='text-embedding-ada-002')
        self.chunk_size = 256
        self.chunk_overlap = 100
        self.service_context = ServiceContext.from_defaults(embed_model=self.embed_model, chunk_size=self.chunk_size, chunk_overlap = self.chunk_overlap)

        integration_token = os.getenv("NOTION_INTEGRATION_TOKEN")
        NotionPageReader = download_loader('NotionPageReader')
        page_ids = ["87b45a97333c40eea3d53c1ea8f983cb"]
        reader = NotionPageReader(integration_token=integration_token)
        documents = reader.load_data(page_ids=page_ids)

        self.index = GPTVectorStoreIndex.from_documents(documents,
                                                        storage_context=self.storage_context,
                                                        service_context=self.service_context)

    def get_response(self, question):
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
        query_engine = self.index.as_query_engine(text_qa_template=qa_template,
                                                  response_mode='compact', llm=OpenAI(model="gpt-3.5-turbo"), similarity_top_k=2)

        start_time = time.time()
        res: Response = query_engine.query(question)
        print(f"time taken: {(time.time() - start_time)}seconds")
        print(res)
        return res.response
    

    def process(self, user_query):
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
        query_engine = self.index.as_query_engine(text_qa_template=qa_template,
                                                  response_mode='refine', llm=OpenAI(model="gpt-3.5-turbo"))

        start_time = time.time()
        res: Response = query_engine.query(email_question)
        print(f"time taken: {(time.time() - start_time)}seconds")
        print(res)
        return res.response
