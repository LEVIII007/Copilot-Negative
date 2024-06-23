from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from prompts import context, code_parser_template
from code_reader import code_reader
from dotenv import load_dotenv
import os
import ast
import requests

load_dotenv()

llm = Ollama(model="mistral", request_timeout=3600.0)

parser = LlamaParse(result_type="markdown")

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="this gives documentation about code for an API. Use this for reading docs for the API",
        ),
    ),
    code_reader,
]

code_llm = Ollama(model="codellama", request_timeout=3600.0)
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

# try:s
#     response = requests.get('http://localhost:8080/api/chat')
#     if response.status_code == 200:
#         print("Successfully connected to the server. Response:", response.text)
#     else:
#         print("Failed to connect to the server. Status code:", response.status_code)
# except requests.exceptions.RequestException as e:
#     print("An error occurred while trying to connect to the server:", e)


class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str


parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template)
print("Debug: Formatted prompt template.")  # Debug statement

json_prompt_tmpl = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])
print("Debug: Output pipeline setup complete.")  # Debug statement

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0

    while retries < 3:
        try:
            print(f"Debug: Sending query '{prompt}' to agent.")  # Debug statement
            result = agent.query(prompt)
            print("Debug: Query sent successfully, processing result.")  # Debug statement

            next_result = output_pipeline.run(response=result)
            print("Debug: Output pipeline run complete.")  # Debug statement

            cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
            print("Debug: Cleaned JSON obtained.")  # Debug statement
            break
        except Exception as e:
            retries += 1
            print(f"Error occurred, retry #{retries}:", e)
            print("Debug: Retrying query due to error.")  # Debug statement

    if retries >= 3:
        print("Unable to process request, try again...")
        continue

    print("Code generated")
    print(cleaned_json["code"])
    print("\n\nDescription:", cleaned_json["description"])

    filename = cleaned_json["filename"]

    try:
        print(f"Debug: Attempting to save file '{filename}'")  # Debug statement
        with open(os.path.join("output", filename), "w") as f:
            f.write(cleaned_json["code"])
        print(f"Debug: File '{filename}' saved successfully.")  # Debug statement
    except Exception as e:
        print("Error saving file...", e)
        print(f"Debug: Failed to save file '{filename}'.")  # Debug statement