import dspy
import dotenv
import os
import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import requests
from dspy.utils import download

class DataConfig:
    """데이터 설정"""
    DATASETS = {
        'qa': {
            'url': 'https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_corpus.jsonl',
            'local_path': 'data/qa_dataset.jsonl',
            'desc': '기술 관련 질문-답변 데이터셋'
        }
    }

class DataLoaderSignature(dspy.Signature):
    """데이터 로더 시그니처"""
    dataset_name = dspy.InputField(desc="데이터셋 이름")
    data_path = dspy.InputField(desc="데이터 파일 경로")
    processed_data = dspy.OutputField(desc="전처리된 데이터")
    statistics = dspy.OutputField(desc="데이터 통계 정보")

class CustomDataLoader(dspy.Module):
    """커스텀 ��이터 로더"""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(DataLoaderSignature)
        self.datasets = {}
        self.ensure_data_directory()
    
    def ensure_data_directory(self) -> None:
        """데이터 디렉토리 생성"""
        Path('data').mkdir(exist_ok=True)
    
    def download_dataset(self, dataset_key: str) -> str:
        """데이터셋 다운로드"""
        try:
            config = DataConfig.DATASETS[dataset_key]
            local_path = config['local_path']
            
            # 디렉토리 생성
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 파일이 없거나 비어있으면 다시 다운로드
            if not Path(local_path).exists() or Path(local_path).stat().st_size == 0:
                print(f"📥 다운로드 시작: {dataset_key}")
                
                # requests를 사용하여 파일 다운로드
                response = requests.get(config['url'])
                response.raise_for_status()
                
                # 응답 내용 확인
                content = response.text.strip()
                if not content:
                    raise ValueError("다운로드된 컨텐츠가 비어있습니다")
                
                # 파일 저장
                with open(local_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ 다운로드 완료: {local_path}")
                
                # 파일 크기 확인
                file_size = Path(local_path).stat().st_size
                print(f"📦 파일 크기: {file_size/1024:.2f} KB")
                
                # 파일 내용 미리보기
                with open(local_path, 'r', encoding='utf-8') as f:
                    preview = f.readline().strip()
                print(f"👀 첫 번째 라인 미리보기: {preview[:100]}...")
            
            else:
                file_size = Path(local_path).stat().st_size
                print(f"📁 기존 파일 사용: {local_path} (크기: {file_size/1024:.2f} KB)")
            
            return local_path
        
        except Exception as e:
            print(f"❌ 다운로드 실패 ({dataset_key}): {str(e)}")
            # 실패한 경우 파일 삭제
            if Path(local_path).exists():
                Path(local_path).unlink()
            raise
    
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """JSONL 파일 로드"""
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if line:  # 빈 라인 무시
                            data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 라인 {i} JSON 파싱 오류: {str(e)}")
                        print(f"문제의 라인: {line[:100]}...")
                        continue
            
            if not data:
                # 파일 내용 확인
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"📄 파일 내용 미리보기:")
                print(content[:500])
                raise ValueError("데이터를 로드할 수 없습니다: 파일이 비어있거나 형식이 잘못되었습니다")
            
            print(f"\n✅ 총 {len(data)}개의 항목을 로드했습니다")
            print(f"📝 첫 번째 항목 미리보기:")
            print(json.dumps(data[0], indent=2, ensure_ascii=False)[:200])
            
            return data
        
        except Exception as e:
            print(f"❌ JSONL 파일 로드 실패 ({file_path}): {str(e)}")
            raise
    
    def preprocess_data(self, data: List[Dict], dataset_key: str) -> pd.DataFrame:
        """데이터 전처리"""
        df = pd.DataFrame(data)
        
        # 데이터셋별 전처리 로직
        if dataset_key == 'qa':
            df = self.preprocess_qa_dataset(df)
        elif dataset_key == 'wiki':
            df = self.preprocess_wiki_dataset(df)
        
        return df
    
    def preprocess_qa_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """QA 데이터셋 전처리"""
        try:
            # 데이터 구조 확인 및 로깅
            print("\n📊 데이터 컬럼:", df.columns.tolist())
            
            # 필요한 컬럼이 없는 경우 기본 구조 생성
            if 'text' not in df.columns:
                # 데이터프레임의 첫 번째 행 출력하여 구조 확인
                print("\n🔍 첫 번째 데이터 샘플:")
                print(df.iloc[0] if not df.empty else "데이터가 비어있습니다")
                
                # 데이터가 단일 텍스 컬럼으로 되어있다면 'text' 컬럼으로 변환
                if len(df.columns) == 1:
                    df = df.rename(columns={df.columns[0]: 'text'})
                else:
                    # 또는 모든 컬럼을 합쳐서 text 컬럼 생성
                    df['text'] = df.apply(lambda row: str(row.to_dict()), axis=1)
            
            # 텍스트 길이 제한
            df['text'] = df['text'].astype(str).str[:6000]
            
            # 중복 제거
            df = df.drop_duplicates(subset=['text'])
            
            return df
        
        except Exception as e:
            print(f"❌ QA 데이터셋 전처리 실패: {str(e)}")
            print(f"현재 데이터프레임 정보:")
            print(df.info())
            raise
    
    def preprocess_wiki_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wikipedia 데이터셋 전처리"""
        # 예시 전처리 로직
        df = df.dropna()
        return df
    
    def calculate_statistics(self, df: pd.DataFrame, dataset_key: str) -> Dict:
        """데이터 통�� 계산"""
        stats = {
            'dataset_name': dataset_key,
            'total_rows': len(df),
            'columns': list(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'null_counts': df.isnull().sum().to_dict()
        }
        return stats
    
    def load_dataset(self, dataset_key: str) -> Dict:
        """데이터셋 로드 및 처리"""
        try:
            # 데이터셋 다운로드
            file_path = self.download_dataset(dataset_key)
            
            # 데이터 로드
            raw_data = self.load_jsonl(file_path)
            
            # 데이터 전처리
            processed_df = self.preprocess_data(raw_data, dataset_key)
            
            # 통계 계산
            statistics = self.calculate_statistics(processed_df, dataset_key)
            
            # 결과 저장
            self.datasets[dataset_key] = {
                'data': processed_df,
                'statistics': statistics
            }
            
            return {
                'dataset_key': dataset_key,
                'data': processed_df,
                'statistics': statistics
            }
        
        except Exception as e:
            print(f"❌ 데이터 로드 실패 ({dataset_key}): {str(e)}")
            raise

def display_dataset_info(dataset_info: Dict) -> None:
    """데이터셋 정보 출력"""
    print("\n" + "="*50)
    print(f"📊 데이터셋: {dataset_info['dataset_key']}")
    print(f"📝 설명: {DataConfig.DATASETS[dataset_info['dataset_key']]['desc']}")
    
    print("\n📈 통계 정보:")
    stats = dataset_info['statistics']
    print(f"- 총 행 수: {stats['total_rows']:,}")
    print(f"- 컬럼: {', '.join(stats['columns'])}")
    print(f"- 메모리 사용량: {stats['memory_usage']:.2f} MB")
    
    print("\n🔍 데이터 미리보기:")
    print(dataset_info['data'].head())
    print("="*50)

def main():
    try:
        # 환경 설정
        dotenv.load_dotenv()
        
        # 데이터 로더 초기화
        loader = CustomDataLoader()
        
        # QA 데이터셋만 처리 (에러 없는 데이터셋)
        dataset_key = 'qa'
        print(f"\n🔄 데이터셋 처리 시작: {dataset_key}")
        dataset_info = loader.load_dataset(dataset_key)
        display_dataset_info(dataset_info)
    
    except Exception as e:
        print(f"❌ 메인 프로세스 에러: {str(e)}")
        raise

if __name__ == "__main__":
    main() 