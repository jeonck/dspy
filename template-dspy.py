import dspy
import dotenv
import os

def setup_environment():
    """환경 설정 및 LM 초기화"""
    dotenv.load_dotenv()
    lm = dspy.LM('openai/gpt-4-turbo', api_key=os.getenv('OPENAI_API_KEY'))
    dspy.configure(lm=lm)

class QuestionAnswerSignature(dspy.Signature):
    """질문-답변을 위한 시그니처"""
    question = dspy.InputField(desc="사용자의 질문")
    context = dspy.InputField(desc="관련 배경 정보나 컨텍스트")
    detailed_answer = dspy.OutputField(desc="상세한 답변")
    summary = dspy.OutputField(desc="답변의 요약")

class TemplateBasedQA(dspy.Module):
    """템플릿 기반 질문-답변 모듈"""
    
    def __init__(self):
        super().__init__()
        
        # Predict 모듈에 프롬프트 템플릿 적용
        self.generate_detailed = dspy.Predict(QuestionAnswerSignature)
        self.generate_detailed.instructions = """
        질문: ${question}
        
        참고 정보: ${context}
        
        다음 형식으로 답변해주세요:
        1. 주요 포인트 분석
        2. 상세 설명
        3. 실제 적용 예시
        4. 결론
        """
        
        self.generate_summary = dspy.Predict(QuestionAnswerSignature)
        self.generate_summary.instructions = """
        다음 상세 답변을 2-3문장으로 요약해주세요:
        ${detailed_answer}
        """
    
    def forward(self, question, context=""):
        # 상세 답변 생성
        detailed_response = self.generate_detailed(
            question=question,
            context=context
        )
        
        # 요약 생성
        summary_response = self.generate_summary(
            question=question,
            context=context,
            detailed_answer=detailed_response.detailed_answer
        )
        
        return {
            'question': question,
            'detailed_answer': detailed_response.detailed_answer,
            'summary': summary_response.summary
        }

def process_query(qa_module, query: str, context: str = "") -> None:
    """쿼리 처리 및 결과 출력"""
    try:
        result = qa_module(question=query, context=context)
        
        print(f"\n❓ 질문:")
        print(result['question'])
        
        if context:
            print(f"\n📚 참고 정보:")
            print(context)
        
        print(f"\n💭 상세 답변:")
        print(result['detailed_answer'])
        
        print(f"\n📌 요약:")
        print(result['summary'])
        
        print("\n" + "="*50)
    
    except Exception as e:
        print(f"Error processing query '{query}': {e}")
        raise

def main():
    try:
        # 환경 설정
        setup_environment()
        
        # QA 모듈 초기화
        qa_module = TemplateBasedQA()
        
        # 테스트 케이스들
        test_cases = [
            {
                "question": "기계학습(Machine Learning)의 주요 종류와 특징은 무엇인가요?",
                "context": """
                기계학습은 크게 지도학습, 비지도학습, 강화학습으로 나눌 수 있습니다.
                각각의 학습 방식은 데이터의 특성과 문제 해결 방식에 따라 선택됩니다.
                최근에는 준지도학습과 자기지도학습 등 새로운 방식도 등장하고 있습니다.
                """
            },
            {
                "question": "클라우드 네이티브 아키텍처란 무엇인가요?",
                "context": """
                클라우드 네이티브는 클라우드 컴퓨팅 모델의 장점을 극대화하도록 설계된 애플리케이션 아키텍처입니다.
                마이크로서비스, 컨테이너화, DevOps 등의 현대적인 방법론을 포함합니다.
                확장성, 유연성, 복원력이 핵심 특징입니다.
                """
            }
        ]
        
        # 각 테스트 케이스 처리
        for test_case in test_cases:
            process_query(qa_module, 
                        query=test_case["question"], 
                        context=test_case["context"])
    
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 