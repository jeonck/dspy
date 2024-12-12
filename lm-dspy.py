import dspy
import dotenv
import os
from typing import Dict, List

def setup_environment():
    """환경 설정 로드"""
    dotenv.load_dotenv()

class ModelConfig:
    """언어 모델 설정"""
    MODELS = {
        'gpt4': {
            'name': 'openai/gpt-4-turbo',
            'desc': 'GPT-4 Turbo - 가장 강력한 추론 능력'
        },
        'gpt35': {
            'name': 'openai/gpt-3.5-turbo',
            'desc': 'GPT-3.5 Turbo - 빠른 응답과 비용 효율'
        },
        'claude': {
            'name': 'anthropic/claude-3-sonnet',
            'desc': 'Claude 3 Sonnet - 학술적 분석에 강점'
        }
    }

class LanguageModelSignature(dspy.Signature):
    """언어 모델 입출력 시그니처"""
    prompt = dspy.InputField(desc="사용자의 입력 프롬프트")
    model_name = dspy.InputField(desc="사용할 모델의 이름")
    response = dspy.OutputField(desc="모델의 응답")
    analysis = dspy.OutputField(desc="응답에 대한 분석")

class MultiModelProcessor(dspy.Module):
    """여러 언어 모델을 처리하는 모듈"""
    
    def __init__(self):
        super().__init__()
        self.models: Dict[str, dspy.LM] = {}
        self.predictor = dspy.Predict(LanguageModelSignature)
    
    def initialize_model(self, model_key: str) -> None:
        """특정 모델 초기화"""
        try:
            model_config = ModelConfig.MODELS[model_key]
            model = dspy.LM(
                model_config['name'],
                api_key=os.getenv('OPENAI_API_KEY')
            )
            self.models[model_key] = model
            print(f"✅ 모델 초기화 완료: {model_config['name']}")
        except Exception as e:
            print(f"❌ 모델 초기화 실패 ({model_key}): {str(e)}")
    
    def initialize_all_models(self) -> None:
        """모든 모델 초기화"""
        for model_key in ModelConfig.MODELS:
            self.initialize_model(model_key)
    
    def process_with_model(self, prompt: str, model_key: str) -> Dict:
        """특정 모델로 프롬프트 처리"""
        if model_key not in self.models:
            self.initialize_model(model_key)
        
        dspy.configure(lm=self.models[model_key])
        
        result = self.predictor(
            prompt=prompt,
            model_name=ModelConfig.MODELS[model_key]['name']
        )
        
        return {
            'model': model_key,
            'model_name': ModelConfig.MODELS[model_key]['name'],
            'model_desc': ModelConfig.MODELS[model_key]['desc'],
            'prompt': prompt,
            'response': result.response,
            'analysis': result.analysis
        }
    
    def process_with_all_models(self, prompt: str) -> List[Dict]:
        """모든 모델로 프롬프트 처리"""
        results = []
        for model_key in ModelConfig.MODELS:
            try:
                result = self.process_with_model(prompt, model_key)
                results.append(result)
            except Exception as e:
                print(f"❌ 처리 실패 ({model_key}): {str(e)}")
        return results

def display_results(results: List[Dict]) -> None:
    """결과 출력"""
    for result in results:
        print("\n" + "="*50)
        print(f"🤖 모델: {result['model_name']}")
        print(f"📝 설명: {result['model_desc']}")
        print("\n📄 프롬프트:")
        print(result['prompt'])
        print("\n💭 응답:")
        print(result['response'])
        print("\n🔍 분석:")
        print(result['analysis'])
        print("="*50)

def main():
    try:
        # 환경 설정
        setup_environment()
        
        # 프로세서 초기화
        processor = MultiModelProcessor()
        
        # 테스트 프롬프트들
        test_prompts = [
            """
            다음 주제에 대해 분석해주세요:
            '인공지능이 현대 사회에 미치는 영향과 윤리적 고려사항'
            """,
            """
            다음 코드의 시간 복잡도를 분석해주세요:
            def fibonacci(n):
                if n <= 1:
                    return n
                return fibonacci(n-1) + fibonacci(n-2)
            """
        ]
        
        # 각 프롬프트에 대해 모든 모델로 처리
        for prompt in test_prompts:
            print("\n🔄 새로운 프롬프트 처리 시작")
            results = processor.process_with_all_models(prompt)
            display_results(results)
            print("\n" + "="*50)
    
    except Exception as e:
        print(f"❌ 메인 프로세스 에러: {str(e)}")
        raise

if __name__ == "__main__":
    main() 