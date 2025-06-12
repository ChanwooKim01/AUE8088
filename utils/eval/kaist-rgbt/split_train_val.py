import os
from sklearn.model_selection import KFold

# 파일 경로
input_txt = "train-all-04.txt"  # 입력 파일명
output_dir = "."  # 결과 파일 저장 경로

# 데이터 읽기
with open(input_txt, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# 5-fold split
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(lines)):
    train_lines = [lines[i] for i in train_idx]
    val_lines = [lines[i] for i in val_idx]

    # 파일로 저장
    with open(os.path.join(output_dir, f"train_fold{fold+1}.txt"), "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(output_dir, f"val_fold{fold+1}.txt"), "w") as f:
        f.write("\n".join(val_lines) + "\n")

print("5-fold split 완료!")