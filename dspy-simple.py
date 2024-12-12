import dspy
import dotenv
import os
dotenv.load_dotenv()

# OpenAI의 gpt-4o-mini 모델을 설정합니다.
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
dspy.configure(lm=lm)

# 질문에 대한 답변을 생성하는 모듈을 정의합니다.
class AnswerQuestion(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

# Chain of Thought 방식을 사용하여 답변을 생성하는 모듈을 설정합니다.
answer_module = dspy.ChainOfThought(AnswerQuestion)

# 예시 질문을 입력하여 답변을 생성합니다.
question = "대한민국의 수도는 어디인가요?"
prediction = answer_module(question=question)

print(f"질문: {question}")
print(f"답변: {prediction.answer}")
