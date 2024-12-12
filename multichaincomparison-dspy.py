import dspy
import dotenv
import os

def setup_environment():
    """환경 설정 및 LM 초기화"""
    dotenv.load_dotenv()
    lm = dspy.LM('openai/gpt-4-turbo', api_key=os.getenv('OPENAI_API_KEY'))
    dspy.configure(lm=lm)

class MathProblemSignature(dspy.Signature):
    """수학 문제 해결을 위한 시그니처"""
    question = dspy.InputField(desc="수학 문제")
    solution_steps = dspy.OutputField(desc="문제 해결 과정")
    final_answer = dspy.OutputField(desc="최종 답안")

class MathProblemSolver(dspy.Module):
    """수학 문제 해결 모듈"""
    
    def __init__(self):
        super().__init__()
        # 여러 해결 방법을 비교하는 MultiChainComparison 설정
        self.predictor = dspy.Predict(MathProblemSignature)
        self.solver = dspy.MultiChainComparison(
            MathProblemSignature,
            M=3,  # 3가지 다른 접근 방식 시도
            temperature=0.7
        )
    
    def generate_completion(self, question: str) -> dspy.Prediction:
        """단일 해결 방법 생성"""
        return self.predictor(
            question=question,
            solution_steps=dspy.ChainOfThought(f"""
            1단계: 문제 이해
            2단계: 해결 방법 계획
            3단계: 계산 수행
            4단계: 답안 검증
            """),
            final_answer="계산 결과에 따른 최종 답안"
        )
    
    def forward(self, question):
        # 문제 전처리
        question = question.strip()
        
        # 여러 해결 방법 생성
        completions = []
        for i in range(3):
            try:
                completion = self.generate_completion(question)
                if completion and hasattr(completion, 'solution_steps'):
                    completions.append(completion)
            except Exception as e:
                print(f"⚠️ 해결 방법 {i+1} 생성 실패: {str(e)}")
        
        if not completions:
            raise ValueError("유효한 해결 방법을 생성할 수 없습니다.")
        
        # MultiChainComparison을 사용하여 최적의 해결책 선택
        result = self.solver(completions, question=question)
        
        return {
            'question': question,
            'solution_steps': result.solution_steps,
            'final_answer': result.final_answer,
            'rationale': getattr(result, 'rationale', '최적의 해결 방법으로 선택되었습니다.')
        }

def process_math_problem(solver, problem: str) -> None:
    """수학 문제 처리 및 결과 출력"""
    try:
        print("\n" + "="*50)
        print(f"📝 문제:")
        print(problem.strip())
        
        result = solver(question=problem)
        
        print(f"\n🤔 해결 과정:")
        print(result['solution_steps'])
        
        print(f"\n✅ 최종 답안:")
        print(result['final_answer'])
        
        print(f"\n📊 선택 근거:")
        print(result['rationale'])
        print("="*50)
    
    except Exception as e:
        print(f"❌ 처리 오류: {str(e)}")
        print("="*50)

def main():
    try:
        # 환경 설정
        setup_environment()
        
        # 수학 문제 해결기 초기화
        solver = MathProblemSolver()
        
        # 테스트할 수학 문제들
        test_problems = [
            "한 가게에서 사과 3개에 1,500원을 받습니다. 영희가 사과 5개를 사려면 얼마를 내야 하나요?",
            "직사각형의 가로 길이가 8cm이고 세로 길이가 5cm입니다. 이 직사각형의 넓이와 둘레를 구하세요.",
            "어떤 수에 7을 더하면 15가 됩니다. 이 수를 구하세요."
        ]
        
        # 각 문제 처리
        for problem in test_problems:
            process_math_problem(solver, problem)
    
    except Exception as e:
        print(f"❌ 메인 프로세스 에러: {str(e)}")
        raise

if __name__ == "__main__":
    main() 