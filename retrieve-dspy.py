# pip install -U dspy-ai ujson rank_bm25

import dspy
import dotenv
import os
import ujson
from dspy.utils import download
from dspy.retrieve import *


def setup_environment():
    """환경 설정 및 LM 초기화"""
    dotenv.load_dotenv()
    lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
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

class RAG(dspy.Module):
    """검색 증강 생성(RAG) 모듈"""
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.respond = dspy.ChainOfThought('context, question -> response')
    
    def forward(self, question):
        # retriever를 사용하여 검색
        context = self.retriever(question)
        return self.respond(context=context, question=question)

def process_query(rag_module, query: str) -> None:
    """쿼리 처리 및 결과 출력"""
    try:
        result = rag_module(question=query)
        
        print(f"\n🔍 검색 쿼리: {query}")
        print("\n💭 생성된 답변:")
        print(result.answer)
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
        
        # RAG 모듈 초기화 (retriever 전달)
        rag = RAG(retriever)
        
        # 테스트 쿼리
        test_queries = [
            "인공지능이란 무엇인가요?",
            "리눅스의 메모리 관리 방식은?",
            "깃허브에서 이미지 파일을 관리하는 방법은?"
        ]
        
        # 각 쿼리에 대해 처리
        for query in test_queries:
            process_query(rag, query)
    
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 