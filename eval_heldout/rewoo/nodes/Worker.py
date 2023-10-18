import requests
from geopy.geocoders import Nominatim
from langchain import OpenAI, LLMMathChain, LLMChain, PromptTemplate, Wikipedia
from langchain.llms import OpenAIChat
from langchain.agents import Tool
from langchain.agents.react.base import DocstoreExplorer
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
# from langchain.utilities import SerpAPIWrapper
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

import sys

from nodes.Node import Node

from fastchat.model.model_adapter import get_conversation_template

from nodes.LLMNode import llm_llama

class GoogleWorker(Node):
    def __init__(self, name="Google"):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = False
        self.description = "Worker that searches results from Google. Useful when you need to find short " \
                           "and succinct answers about a specific topic. Input should be a search query."

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        # tool = SerpAPIWrapper()
        # import IPython
        # IPython.embed()
        tool = GoogleSerperAPIWrapper()
        evidence = tool.run(input)
        assert isinstance(evidence, self.output_type)
        if log:
            print(f"Running {self.name} with input {input}\nOutput: {evidence}\n")
        return evidence


class WikipediaWorker(Node):
    def __init__(self, name="Wikipedia", docstore=None):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = False
        self.description = "Worker that search for similar page contents from Wikipedia. Useful when you need to " \
                           "get holistic knowledge about people, places, companies, historical events, " \
                           "or other subjects. The response are long and might contain some irrelevant information. " \
                           "Input should be a search query."
        self.docstore = docstore

    def run(self, input, log=False):
        if not self.docstore:
            self.docstore = DocstoreExplorer(Wikipedia())
        assert isinstance(input, self.input_type)
        tool = Tool(
            name="Search",
            func=self.docstore.search,
            description="useful for when you need to ask with search"
        )
        evidence = tool.run(input)
        assert isinstance(evidence, self.output_type)
        if log:
            print(f"Running {self.name} with input {input}\nOutput: {evidence}\n")
        return evidence


class DocStoreLookUpWorker(Node):
    def __init__(self, name="LookUp", docstore=None):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = False
        self.description = "Worker that search the direct sentence in current Wikipedia result page. Useful when you " \
                           "need to find information about a specific keyword from a existing Wikipedia search " \
                           "result. Input should be a search keyword."
        self.docstore = docstore

    def run(self, input, log=False):
        if not self.docstore:
            raise ValueError("Docstore must be provided for lookup")
        assert isinstance(input, self.input_type)
        tool = Tool(
            name="Lookup",
            func=self.docstore.lookup,
            description="useful for when you need to ask with lookup"
        )
        evidence = tool.run(input)
        assert isinstance(evidence, self.output_type)
        if log:
            print(f"Running {self.name} with input {input}\nOutput: {evidence}\n")
        return evidence


class CustomWolframAlphaAPITool(WolframAlphaAPIWrapper):
    def __init__(self):
        super().__init__()

    def run(self, query: str) -> str:
        """Run query through WolframAlpha and parse result."""
        res = self.wolfram_client.query(query)

        try:
            answer = next(res.results).text
        except StopIteration:
            return "Wolfram Alpha wasn't able to answer it"

        if answer is None or answer == "":
            return "No good Wolfram Alpha Result was found"
        else:
            return f"Answer: {answer}"


class WolframAlphaWorker(Node):
    def __init__(self, name="WolframAlpha"):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = False
        self.description = "A WolframAlpha search engine. Useful when you need to solve a complicated Mathematical or " \
                           "Algebraic equation. Input should be an equation or function."

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        tool = CustomWolframAlphaAPITool()
        evidence = tool.run(input).replace("Answer:", "").strip()
        assert isinstance(evidence, self.output_type)
        if log:
            print(f"Running {self.name} with input {input}\nOutput: {evidence}\n")
        return evidence


class CalculatorWorker(Node):
    def __init__(self, name="Calculator"):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = True
        self.description = "A calculator that can compute arithmetic expressions. Useful when you need to perform " \
                           "math calculations. Input should be a mathematical expression"

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        llm = OpenAI(temperature=0)
        tool = LLMMathChain(llm=llm, verbose=False)
        response = tool(input)
        evidence = response["answer"].replace("Answer:", "").strip()
        assert isinstance(evidence, self.output_type)
        if log:
            return {"input": response["question"], "output": response["answer"]}
        return evidence


class LLMWorker(Node):
    def __init__(self, name="LLM", model='gpt-3.5-turbo'):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = True
        self.model = model
        self.description = "A pretrained LLM like yourself. Useful when you need to act with general world " \
                           "knowledge and common sense. Prioritize it when you are confident in solving the problem " \
                           "yourself. Input can be any instruction."

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        if self.model in {'gpt-3.5-turbo', 'gpt-4'}:
            llm = OpenAIChat(temperature=0, model=self.model)
            prompt = PromptTemplate(template="Respond in short directly with no extra words.\n\n{request}",
                                    input_variables=["request"])
            tool = LLMChain(prompt=prompt, llm=llm, verbose=False)
            response = tool(input)
            evidence = response["text"].strip("\n")
            assert isinstance(evidence, self.output_type)
            if log:
                return {"input": response["request"], "output": response["text"]}
            return evidence
        else:
            conv = get_conversation_template('llama-2')
            conv.set_system_message("You are a helpful, respectful and honest assistant.")
            conv.append_message(conv.roles[0], f'Responds in short directly with no extra words.\n\n{input}')
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            response = llm_llama(prompt)
            evidence = response.strip('\n')
            assert isinstance(evidence, self.output_type)
            if log:
                return {"input": prompt, "output": evidence}
            return evidence


class ZipCodeRetriever(Node):

    def __init__(self, name="ZipCodeRetriever"):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = False
        self.description = "A zip code retriever. Useful when you need to get users' current zip code. Input can be " \
                           "left blank."

    def get_ip_address(self):
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        return data["ip"]

    def get_location_data(sefl, ip_address):
        url = f"https://ipinfo.io/{ip_address}/json"
        response = requests.get(url)
        data = response.json()
        return data

    def get_zipcode_from_lat_long(self, lat, long):
        geolocator = Nominatim(user_agent="zipcode_locator")
        location = geolocator.reverse((lat, long))
        return location.raw["address"]["postcode"]

    def get_current_zipcode(self):
        ip_address = self.get_ip_address()
        location_data = self.get_location_data(ip_address)
        lat, long = location_data["loc"].split(",")
        zipcode = self.get_zipcode_from_lat_long(float(lat), float(long))
        return zipcode

    def run(self, input):
        assert isinstance(input, self.input_type)
        evidence = self.get_current_zipcode()
        assert isinstance(evidence, self.output_type)


class SearchDocWorker(Node):

    def __init__(self, doc_name, doc_path, name="SearchDoc"):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = True
        self.doc_path = doc_path
        self.description = f"A vector store that searches for similar and related content in document: {doc_name}. " \
                           f"The result is a huge chunk of text related to your search but can also " \
                           f"contain irrelevant info. Input should be a search query."

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        loader = TextLoader(self.doc_path)
        vectorstore = VectorstoreIndexCreator().from_loaders([loader]).vectorstore
        evidence = vectorstore.similarity_search(input, k=1)[0].page_content
        assert isinstance(evidence, self.output_type)
        if log:
            print(f"Running {self.name} with input {input}\nOutput: {evidence}\n")
        return evidence


class SearchSOTUWorker(SearchDocWorker):
    def __init__(self, name="SearchSOTU"):
        super().__init__(name=name, doc_name="state_of_the_union", doc_path="data/docs/state_of_the_union.txt")



WORKER_REGISTRY = {"Google": GoogleWorker(),
                   "Wikipedia": WikipediaWorker(),
                   "LookUp": DocStoreLookUpWorker(),
                   "WolframAlpha": WolframAlphaWorker(),
                   "Calculator": CalculatorWorker(),
                   "LLM": LLMWorker(),
                   "SearchSOTU": SearchSOTUWorker()}
