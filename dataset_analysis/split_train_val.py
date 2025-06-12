# 파일 이름 정의
input_file = "/home/chanwoo/git/deeplearning_2025_1/AUE8088/datasets/kaist-rgbt/train-all-04.txt"  # 원본 txt 파일 이름
train_file = "/home/chanwoo/git/deeplearning_2025_1/AUE8088/datasets/kaist-rgbt/train-set.txt"  # set00, set02, set04로 시작하는 파일 저장
val_file = "/home/chanwoo/git/deeplearning_2025_1/AUE8088/datasets/kaist-rgbt/val-set.txt"  # set01, set03, set05로 시작하는 파일 저장

# 파일 읽기 및 분류
with open(input_file, "r") as infile, \
     open(train_file, "w") as train_out, \
     open(val_file, "w") as val_out:
    for line in infile:
        # 줄에서 setXX 부분 추출
        parts = line.strip().split("/")
        if len(parts) > 2:
            set_name = parts[-1].split("_")[0]  # setXX 추출
            # 조건에 따라 파일에 저장
            if set_name in ["set00", "set02", "set04"]:
                train_out.write(line)
            elif set_name in ["set01", "set03", "set05"]:
                val_out.write(line)

print("파일 분류 완료!")