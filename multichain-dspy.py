import dspy
import dotenv
import os

def setup_environment():
    """환경 설정 및 LM 초기화"""
    dotenv.load_dotenv()
    lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
    dspy.configure(lm=lm)

class AnswerGenerator(dspy.Signature):
    """질문에 대한 답변을 생성합니다."""
    question: str = dspy.InputField(desc="사용자의 질문")
    reasoning: str = dspy.OutputField(desc="답변에 대한 추론 과정")
    answer: str = dspy.OutputField(desc="생성된 답변")

class AnswerValidator(dspy.Signature):
    """생성된 답변의 품질을 평가합니다."""
    question: str = dspy.InputField(desc="원래 질문")
    candidate_answer: str = dspy.InputField(desc="생성된 답변")
    reasoning: str = dspy.InputField(desc="답변의 추론 과정")
    score: float = dspy.OutputField(desc="답변의 품질 점수 (0-1)")
    explanation: str = dspy.OutputField(desc="평가 설명")

def process_question(question: str) -> None:
    """질문을 처리하고 최적의 답변을 선택"""
    # 생성기와 검증기 생성
    generator = dspy.Predict(AnswerGenerator)
    validator = dspy.Predict(AnswerValidator)
    
    # 여러 답변 생성
    answers = []
    for _ in range(3):  # 3개의 답변 생성
        result = generator(question=question)
        answers.append(result)
    
    # 각 답변 평가
    scored_answers = []
    for ans in answers:
        validation = validator(
            question=question,
            candidate_answer=ans.answer,
            reasoning=ans.reasoning
        )
        scored_answers.append((ans, validation.score))
    
    # 최고 점수의 답변 선택
    best_answer = max(scored_answers, key=lambda x: x[1])
    
    # 결과 출력
    print(f"\n질문: {question}\n")
    print("생성된 답변들:")
    for i, (answer, score) in enumerate(scored_answers, 1):
        print(f"\n답변 {i} (점수: {score:.2f}):")
        print(f"추론: {answer.reasoning}")
        print(f"답변: {answer.answer}")
    
    print(f"\n👑 최적의 답변 (점수: {best_answer[1]:.2f}):")
    print(f"추론: {best_answer[0].reasoning}")
    print(f"답변: {best_answer[0].answer}")
    print("\n" + "="*50)

def main():
    # 환경 설정
    setup_environment()
    
    # 테스트할 질문들
    test_questions = [
        "인공지능이 인간의 일자리를 대체할까요?",
        "기후 변화에 대응하기 위해 개인이 할 수 있는 일은 무엇일까요?",
        "효과적인 학습 방법은 무엇인가요?"
    ]
    
    # 각 질문에 대해 처리
    for question in test_questions:
        process_question(question)

if __name__ == "__main__":
    main() 