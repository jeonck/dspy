import dspy
import dotenv
import os
from typing import List

def setup_environment():
    """환경 설정 및 LM 초기화"""
    dotenv.load_dotenv()
    lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
    dspy.configure(lm=lm)

def create_math_solver():
    """수학 문제 해결을 위한 ChainOfThought 시그니처 정의"""
    class MathProblemSolver(dspy.Signature):
        """수학 문제를 단계별로 해결합니다."""
        question: str = dspy.InputField(desc="수학 문제")
        steps: List[str] = dspy.OutputField(desc="문제 해결 단계")
        answer: str = dspy.OutputField(desc="최종 답안")
    
    return dspy.ChainOfThought(MathProblemSolver)

def solve_math_problems(problems: list[str]) -> None:
    """수학 문제 리스트를 해결하고 결과 출력"""
    solver = create_math_solver()
    
    for problem in problems:
        result = solver(question=problem)
        print(f"\n문제: {problem}")
        print("\n풀이 과정:")
        for i, step in enumerate(result.steps, 1):
            print(f"{i}. {step}")
        print(f"\n답: {result.answer}\n")
        print("-" * 50)

def main():
    # 환경 설정
    setup_environment()
    
    # 테스트할 수학 문제들
    test_problems = [
        "철수가 사과 5개를 가지고 있었습니다. 영희가 3개를 더 주었고, 동생에게 2개를 주었습니다. 철수에게 남은 사과는 몇 개인가요?",
        "한 상자에 연필이 12자루씩 들어있습니다. 학급 30명에게 연필을 2자루씩 나누어주려면 몇 상자가 필요한가요?",
    ]
    
    # 문제 해결 실행
    solve_math_problems(test_problems)

if __name__ == "__main__":
    main()
