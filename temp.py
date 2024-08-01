import faiss
import numpy as np
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import PointStruct

def extract_all_vectors_from_faiss(index_path):
    # FAISS 인덱스 로드
    index = faiss.read_index(index_path)
    
    # 총 벡터 수 확인
    total_vectors = index.ntotal
    
    # 모든 벡터를 저장할 배열 생성
    vectors = np.empty((total_vectors, index.d), dtype=np.float32)
    
    # 모든 벡터 추출
    for i in range(total_vectors):
        vector = index.reconstruct(i)
        vectors[i] = vector
    
    return vectors, index.d

def upload_vectors_to_qdrant(vectors, dimension, qdrant_url, collection_name, api_key=None):
    # Qdrant 클라이언트 생성 (타임아웃 설정 증가)
    qdrant_client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=300)
    
    try:
        # 컬렉션 존재 여부 확인 및 생성
        collections = qdrant_client.get_collections()
        if collection_name not in [col.name for col in collections.collections]:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(size=dimension, distance=rest.Distance.COSINE)
            )
            print(f"컬렉션 '{collection_name}'이 생성되었습니다.")
        else:
            print(f"컬렉션 '{collection_name}'이 이미 존재합니다.")

        # 배치 크기 설정
        batch_size = 100
        total_vectors = len(vectors)

        for i in range(0, total_vectors, batch_size):
            batch = vectors[i:i+batch_size]
            points = [
                PointStruct(id=idx+i, vector=vector.tolist())
                for idx, vector in enumerate(batch)
            ]

            retry_count = 0
            while retry_count < 5:  # 최대 5번 재시도
                try:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    print(f"배치 {i//batch_size + 1}/{total_vectors//batch_size + 1} 업로드 완료")
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"배치 업로드 실패 {retry_count}회. 재시도 중... (오류: {str(e)})")
                    time.sleep(5 * retry_count)  # 재시도 간격을 점진적으로 늘림

            if retry_count == 5:
                print(f"배치 {i//batch_size + 1} 업로드 실패. 다음 배치로 넘어갑니다.")

        print(f"{total_vectors}개의 벡터가 성공적으로 업로드되었습니다.")

    except Exception as e:
        print(f"오류 발생: {str(e)}")

def main():
    # FAISS 인덱스 파일 경로
    faiss_index_path = "db/faiss/index.faiss"
    
    # Qdrant 설정
    qdrant_url = "https://fcfe840e-ae96-464a-8108-4efed0b9d6f6.us-east4-0.gcp.cloud.qdrant.io:6333"  # Qdrant 서버 URL
    collection_name = "dongju"  # 원하는 컬렉션 이름
    api_key = '_OI5LJq0mu3Ai_iSikNWX3lAlfl8m1Wa5788sxEBQpAArflWQ_dywA' # API 키가 필요한 경우 여기에 입력, 불필요시 None

    try:
        # FAISS에서 벡터 추출
        print("FAISS 인덱스에서 벡터 추출 중...")
        vectors, dimension = extract_all_vectors_from_faiss(faiss_index_path)
        print(f"추출된 벡터: {len(vectors)}, 차원: {dimension}")

        # Qdrant에 벡터 업로드
        print("Qdrant에 벡터 업로드 중...")
        upload_vectors_to_qdrant(vectors, dimension, qdrant_url, collection_name, api_key)

    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()