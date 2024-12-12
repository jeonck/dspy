# pip install faiss-cpu

import dspy
import dotenv
import os
import ujson
from dspy.utils import download
from dspy.retrieve import *

def setup_environment():
    """환경 설정 및 LM 초기화"""
    dotenv.load_dotenv()
    lm = dspy.LM('openai/gpt-4-turbo', api_key=os.getenv('OPENAI_API_KEY'))
    dspy.configure(lm=lm)

def setup_retriever():
    """검색기 설정"""
    try:
        # 샘플 코퍼스 다운로드
        download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_corpus.jsonl")
        
        # 코퍼스 로드
        max_characters = 6000
        with open("ragqa_arena_tech_corpus.jsonl") as f:
            corpus = [ujson.loads(line)['text'][:max_characters] for line in f]
            print(f"Loaded {len(corpus)} documents")
        
        # Embeddings 검색기 설정 
        embedder = dspy.Embedder('openai/text-embedding-3-small', dimensions=512)
        retriever = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=3)
        return retriever
    
    except Exception as e:
        print(f"Error in setup_retriever: {e}")
        raise

class GenerateAnswerSignature(dspy.Signature):
    """답변 생성을 위한 시그니처"""
    context = dspy.InputField(desc="검색된 관련 문서들")
    question = dspy.InputField(desc="사용자의 질문")
    thought_process = dspy.OutputField(desc="질문에 답변하기 위한 사고 과정")
    final_answer = dspy.OutputField(desc="생성된 최종 답변")

class RetrieveSignature(dspy.Signature):
    """문서 검색을 위한 시그니처"""
    question = dspy.InputField(desc="사용자의 질문")
    search_query = dspy.OutputField(desc="검색에 사용될 최적화된 쿼리")
    
class RAG(dspy.Module):
    """검색 증강 생성(RAG) 모듈"""
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.generate_query = dspy.Predict(RetrieveSignature)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswerSignature)
    
    def forward(self, question):
        # 검색 쿼리 최적화
        retrieved_query = self.generate_query(question=question)
        
        # retriever를 사용하여 검색
        context = self.retriever(retrieved_query.search_query)
        
        # 답변 생성
        response = self.generate_answer(
            context=context,
            question=question
        )
        
        return {
            'search_query': retrieved_query.search_query,
            'context': context,
            'thought_process': response.thought_process,
            'final_answer': response.final_answer
        }

def process_query(rag_module, query: str) -> None:
    """쿼리 처리 및 결과 출력"""
    try:
        result = rag_module(question=query)
        
        print(f"\n🔍 원본 쿼리: {query}")
        print(f"🔎 최적화된 검색 쿼리: {result['search_query']}")
        print("\n📚 검색된 컨텍스트:")
        print(result['context'])
        print("\n🤔 사고 과정:")
        print(result['thought_process'])
        print("\n💭 최종 답변:")
        print(result['final_answer'])
        print("\n" + "="*50)
    
    except Exception as e:
        print(f"Error processing query '{query}': {e}")
        raise

def main():
    try:
        # 환경 설정
        setup_environment()
        
        # 검색기 설정
        retriever = setup_retriever()
        
        # RAG 모듈 초기화
        rag = RAG(retriever)
        
        # 테스트 쿼리
        test_queries = [
            "인공지능의 윤리적 문제점은 무엇인가요?",
            "블록체인 기술의 실제 활용 사례를 설명해주세요.",
            "클라우드 컴퓨팅의 장단점은?"
        ]
        
        # 각 쿼리에 대해 처리
        for query in test_queries:
            process_query(rag, query)
    
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 