import dspy
import dotenv
import os

def setup_environment():
    """환경 설정 및 LM 초기화"""
    dotenv.load_dotenv()
    lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
    dspy.configure(lm=lm)

def create_algorithm_designer():
    """알고리즘 설계를 위한 ProgramOfThought 시그니처 정의"""
    class AlgorithmDesigner(dspy.Signature):
        """주어진 문제에 대한 알고리즘을 설계하고 구현 단계를 제시합니다."""
        problem: str = dspy.InputField(desc="해결할 문제")
        solution: str = dspy.OutputField(desc="알고리즘 설계 및 구현 상세 정보")
    
    return dspy.ProgramOfThought(AlgorithmDesigner)

def solve_programming_problems(problems: list[str]) -> None:
    """프로그래밍 문제를 해결하고 결과 출력"""
    designer = create_algorithm_designer()
    
    for problem in problems:
        result = designer(problem=problem)
        print(f"\n문제: {problem}\n")
        print("해결 방안:")
        print(result.solution)
        print("\n" + "="*50)

def main():
    # 환경 설정
    setup_environment()
    
    # 테스트할 프로그래밍 문제들
    test_problems = [
        """주어진 정수 배열에서 가장 큰 부분합을 찾는 알고리즘을 설계하세요. 
        예: [-2, 1, -3, 4, -1, 2, 1, -5, 4] -> 답: 6 (부분배열 [4, -1, 2, 1])""",
        
        """두 문자열이 서로 애너그램인지 확인하는 알고리즘을 설계하세요.
        예: "listen"과 "silent"는 애너그램입니다."""
    ]
    
    # 문제 해결 실행
    solve_programming_problems(test_problems)

if __name__ == "__main__":
    main()
