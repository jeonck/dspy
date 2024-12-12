import dspy
import dotenv
import os
from typing import Literal

def setup_environment():
    """환경 설정 및 LM 초기화"""
    dotenv.load_dotenv()
    lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
    dspy.configure(lm=lm)

def create_sentiment_classifier():
    """감정 분류기 시그니처 정의 및 predictor 생성
    
    주요 특징:
    - typing.Literal을 사용하여 가능한 감정 값들을 명시적으로 정의했습니다.
    - dspy.Signature를 상속받는 클래스를 만들어 입력과 출력 필드를 정의했습니다.
    - 입력 필드(text)와 출력 필드(sentiment)를 명확하게 지정했습니다.
    - predictor 호출 시 명명된 매개변수를 사용하도록 했습니다.
    
    Returns:
        dspy.Predict: 설정된 감정 분류기 predictor
    """
    class SentimentClassifier(dspy.Signature):
        """문장의 감정을 분류합니다."""
        text: str = dspy.InputField()
        sentiment: Literal['긍정', '부정', '중립'] = dspy.OutputField()
    
    return dspy.Predict(SentimentClassifier)

def analyze_sentiments(sentences: list[str]) -> None:
    """문장 리스트의 감정을 분석하고 결과 출력"""
    predictor = create_sentiment_classifier()
    
    for sentence in sentences:
        result = predictor(text=sentence)
        print(f"문장: {sentence}")
        print(f"감정: {result.sentiment}\n")

def main():
    # 환경 설정
    setup_environment()
    
    # 테스트할 문장들
    test_sentences = [
        "오늘 날씨가 정말 좋아서 기분이 너무 좋아요!",
        "시험에 떨어져서 너무 실망스럽네요.",
        "오늘 점심으로 김밥을 먹었어요."
    ]
    
    # 감정 분석 실행
    analyze_sentiments(test_sentences)

if __name__ == "__main__":
    main()

