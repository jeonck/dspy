import dspy
import dotenv
import os
import datetime

def setup_environment():
    """환경 설정 및 LM 초기화"""
    dotenv.load_dotenv()
    lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
    dspy.configure(lm=lm)

# 개별 도구 함수들을 정의
def get_current_time():
    """현재 시간을 반환합니다."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str):
    """간단한 수식을 계산합니다."""
    try:
        return eval(expression)
    except:
        return "계산할 수 없는 수식입니다."

def search_weather(city: str):
    """도시의 날씨를 검색합니다 (예시 데이터)."""
    weather_data = {
        "서울": "맑음, 20도",
        "부산": "흐림, 22도",
        "제주": "비, 19도"
    }
    return weather_data.get(city, "날씨 정보를 찾을 수 없습니다.")

def create_assistant():
    """ReAct 기반 어시스턴트 생성"""
    class Assistant(dspy.Signature):
        """사용자 입력에 반응하고 적절한 도구를 사용하여 응답하는 어시스턴트"""
        user_input: str = dspy.InputField(desc="사용자의 입력")
        thoughts: str = dspy.OutputField(desc="상황 분석 및 행동 계획")
        action: str = dspy.OutputField(desc="수행할 작업")
        response: str = dspy.OutputField(desc="최종 응답")
    
    # 도구들을 리스트로 전달
    tools = [get_current_time, calculate, search_weather]
    return dspy.ReAct(Assistant, tools)

def process_user_input(assistant, user_input: str) -> None:
    """사용자 입력을 처리하고 결과 출력"""
    result = assistant(user_input=user_input)
    
    print(f"\n사용자: {user_input}")
    print(f"\n🤔 생각: {result.thoughts}")
    print(f"🛠️ 행동: {result.action}")
    print(f"💬 응답: {result.response}")
    print("\n" + "="*50)

def main():
    # 환경 설정
    setup_environment()
    
    # 어시스턴트 생성
    assistant = create_assistant()
    
    # 테스트할 사용자 입력들
    test_inputs = [
        "지금 몇 시야?",
        "서울 날씨 어때?",
        "123 더하기 456은?",
        "오늘 할 일 정리해줘"
    ]
    
    # 각 입력에 대해 처리
    for user_input in test_inputs:
        process_user_input(assistant, user_input)

if __name__ == "__main__":
    main()
