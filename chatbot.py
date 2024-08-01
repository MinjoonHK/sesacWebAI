import os
import re
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
import os
import faiss


os.environ['KMP_DUPLICATE_LIB_OK']='True'


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# from predict import classify
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor 


# 환경 변수 읽기
openai_api_key = os.getenv('OPENAI_API_KEY')

def load_data(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()


def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200,
        chunk_overlap=50,
        encoding_name='cl100k_base'
    )
    return text_splitter.split_documents(text)


def create_embeddings():
    embeddings_model = OpenAIEmbeddings()
    return embeddings_model


def create_vector_store(documents, embeddings_model):
    vectorstore = FAISS.from_documents(
        documents,
        embedding=embeddings_model,
        distance_strategy=DistanceStrategy.COSINE
    )
    vectorstore.save_local('./db/faiss')
    return vectorstore


def load_vector_store(vectorstore_file: str):
    loaded_vectorstore = FAISS.load_local(vectorstore_file, Embeddings_model, allow_dangerous_deserialization=True)
    return loaded_vectorstore


def retrieve_documents(vectorstore, query, k=4):
    retriever = vectorstore.as_retriever(search_kwargs={'k': k})
    return retriever.get_relevant_documents(query)


def format_documents(docs):
    return '\n\n'.join([d.page_content for d in docs])


def check_positive_negative(query:str):
    return classify(query)

# remove ** in answer.
def process_response(response:str)->str:
    response = response.replace('**', '')
    response = response.replace('\n', '\n ')
    
    return response

# find date in answer
def find_date(response:str):
    dates_list = []
    lines = response.split('\n')
    date_pattern = r'\b\d{4}-\d{2}-\d{2}\b'

    for line in lines:
        matches = re.findall(date_pattern, line)
        if matches:
            for a in matches:
                dates_list.append(a)
                
    if len(dates_list) == 0:
       None 

    return dates_list

def extract_program_info(text):
    programs = []
    lines = text.split('\n')
    current_program = {}
    
    for line in lines:
        if re.match(r'\d+\.', line):
            if current_program:
                programs.append(current_program)
            current_program = {'program': line.split('.', 1)[1].strip().rstrip(':')}
        elif '신청 기한:' in line or '신청 기간:' in line:
            dates = re.findall(r'\d{4}-\d{2}-\d{2} ~ \d{4}-\d{2}-\d{2}', line)

            if len(dates) > 0:
                start, end = dates[0].split(' ~ ')
                current_program['start_date'] = start
                current_program['end_date'] = end
    
    if current_program:
        programs.append(current_program)
    
    return programs

def run_query(vectorstore, query):
    global conversation_history
    retriever = retrieve_documents(vectorstore, query)
    search = TavilySearchResults(k=4)
    
    retriever_tool = create_retriever_tool(
        retriever,
        name = "pdf_search",
        description="지원금, 보조금, 청년를 위한 자립정보는 pdf 문서에서 검색한다. 지원금, 보조금, 청년를 위한 자립정보와 관련된 질문은 이 도구를 사용해야 한다!"            
    )

    tools = [search, retriever_tool]

    # 검색 tool 

    template = '''
    You are named 당찬 and a chatbot service for self-reliance-ready young people!
    Answer the question in Korean based on the information in context and chat_history.
    Be concise in your answer.
    Introduce yourself kindly in the beginning in Korean. 
    Tell kind and intimate.

    Express the format in YYYY-mm-dd when the information about all the dates is in.

    context : {context},
    chat_history : {history},
    Question: {question}
    {agent_scratchpad}
    '''

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(
        temperature=0.0,
        model_name="gpt-4o",  # 모델명
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools = tools, verbose=True )
    
    history = '\n'.join(conversation_history)
        
    response = agent_executor.invoke({'context': format_documents(retriever), 'question': query, 'history':history})
    response = process_response(response['output'])
    conversation_history.append(f'질문 : {query}')
    conversation_history.append(f'답변 : {response}')

    program_info = extract_program_info(response)

    return_dict = {
        'response':response,
        'program_info':program_info
    }
    # if check_positive_negative(query):
    #     print("부정")

    return return_dict


conversation_history = []
file_path = 'Service_detail_api_merged.pdf'
vectorstore_file = './db/faiss'

data = load_data(file_path)
documents = split_text_into_chunks(data)
Embeddings_model = create_embeddings()


# 벡터 쿼리
# query_vector = np.array([your_query_vector], dtype=np.float32)
# if os.path.exists('./db/faiss/index.faiss'):
#     print('load vectorstore')
Vectorstore = load_vector_store('./db/faiss')
    # Vectorstore =  faiss.read_index('./db/faiss/index.faiss')
# else:
#     print('create vectorstore')
#     Vectorstore = create_vector_store(documents, Embeddings_model)

if __name__ == '__main__':
    while True:
        query = input("입력해주세요 : ")
        if query=='q':
            print("exit!")
            break
        print(run_query(Vectorstore, query))

