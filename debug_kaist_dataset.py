#!/usr/bin/env python3
"""
KAIST 데이터셋 라벨 검증 스크립트
에러 원인을 파악하기 위한 데이터 검증
"""

import os
import glob
import yaml
import torch
import numpy as np
from pathlib import Path

def validate_kaist_dataset():
    """KAIST 데이터셋 라벨 검증"""
    
    # 데이터셋 경로 확인
    data_yaml = 'kaist-rgbt.yaml'
    
    print("🔍 KAIST 데이터셋 검증 시작...")
    
    try:
        # 데이터 설정 로드
        with open(data_yaml, 'r') as f:
            data_dict = yaml.safe_load(f)
        
        print(f"✅ 데이터 YAML 로드 성공: {data_yaml}")
        print(f"📊 클래스 수: {data_dict['nc']}")
        print(f"📋 클래스 이름: {data_dict['names']}")
        
        # 각 분할별 검증
        for split in ['train', 'val']:
            if split in data_dict:
                split_path = data_dict[split]
                print(f"\n🔍 {split.upper()} 데이터 검증:")
                print(f"📁 경로: {split_path}")
                
                # 이미지 파일 확인
                if isinstance(split_path, list):
                    # 리스트 형태인 경우
                    total_images = 0
                    for path in split_path:
                        if os.path.exists(path):
                            with open(path, 'r') as f:
                                lines = f.readlines()
                                total_images += len(lines)
                                print(f"  📄 {path}: {len(lines)} 이미지")
                        else:
                            print(f"  ❌ 파일 없음: {path}")
                    print(f"  📊 총 이미지 수: {total_images}")
                else:
                    # 단일 파일인 경우
                    if os.path.exists(split_path):
                        with open(split_path, 'r') as f:
                            lines = f.readlines()
                            print(f"  📊 이미지 수: {len(lines)}")
                            
                            # 처음 몇 개 이미지의 라벨 검증
                            print(f"  🔍 라벨 검증 (처음 10개):")
                            class_counts = {}
                            invalid_labels = []
                            
                            for i, line in enumerate(lines[:10]):
                                img_path = line.strip()
                                label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
                                
                                if os.path.exists(label_path):
                                    try:
                                        with open(label_path, 'r') as lf:
                                            label_lines = lf.readlines()
                                            
                                        for label_line in label_lines:
                                            parts = label_line.strip().split()
                                            if len(parts) >= 5:
                                                class_id = int(parts[0])
                                                if class_id < 0 or class_id >= data_dict['nc']:
                                                    invalid_labels.append(f"{label_path}: class {class_id}")
                                                else:
                                                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                                    except Exception as e:
                                        print(f"    ❌ 라벨 파일 읽기 실패: {label_path} - {e}")
                                else:
                                    print(f"    ⚠️ 라벨 파일 없음: {label_path}")
                            
                            if invalid_labels:
                                print(f"    ❌ 잘못된 클래스 라벨 발견:")
                                for inv_label in invalid_labels[:5]:  # 처음 5개만 표시
                                    print(f"      {inv_label}")
                            else:
                                print(f"    ✅ 라벨 클래스 범위 정상")
                            
                            print(f"    📊 클래스별 객체 수: {class_counts}")
                    else:
                        print(f"  ❌ 파일 없음: {split_path}")
        
    except Exception as e:
        print(f"❌ 검증 중 오류 발생: {e}")
        return False
    
    print("\n✅ 데이터셋 검증 완료")
    return True

def check_gpu_memory():
    """GPU 메모리 상태 확인"""
    if torch.cuda.is_available():
        print(f"\n🖥️ GPU 메모리 상태:")
        print(f"  총 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"  할당된 메모리: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
        print(f"  예약된 메모리: {torch.cuda.memory_reserved() / 1e9:.1f}GB")
    else:
        print("❌ CUDA를 사용할 수 없습니다")

def suggest_safe_training_params():
    """안전한 훈련 파라미터 제안"""
    print(f"\n💡 안전한 훈련 파라미터 제안:")
    print(f"  --batch-size 4    # 배치 크기 감소")
    print(f"  --workers 2       # 워커 수 감소")
    print(f"  --cache ram       # 메모리 캐싱 사용")
    print(f"  --patience 50     # 조기 종료 인내심 감소")

if __name__ == "__main__":
    print("🔧 KAIST 데이터셋 문제 진단 도구")
    print("=" * 50)
    
    # 현재 디렉토리 확인
    print(f"📁 현재 작업 디렉토리: {os.getcwd()}")
    
    # 데이터셋 검증
    validate_kaist_dataset()
    
    # GPU 메모리 확인
    check_gpu_memory()
    
    # 안전한 파라미터 제안
    suggest_safe_training_params()
    
    print("\n🚀 수정된 훈련 명령어:")
    print("python train_simple.py \\")
    print("    --data kaist-rgbt.yaml \\")
    print("    --cfg models/custom_models/mymodel_cbam.yaml \\")
    print("    --hyp data/hyps/hyp.kaist-rgbt.yaml \\")
    print("    --epochs 200 --batch-size 4 --workers 2 \\")
    print("    --weights '' --rgbt --device 0 \\")
    print("    --name kaist_cbam_stable_experiment \\")
    print("    --cache ram --patience 50")
