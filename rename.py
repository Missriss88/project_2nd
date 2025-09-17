import os
import re
from glob import glob

# !!! 경로 확인 필수 !!!
# labels 폴더의 실제 경로를 지정해주세요.
label_dir = os.path.join(os.getcwd(), "resized", "labels")

if not os.path.isdir(label_dir):
    print(f"[오류] 경로를 찾을 수 없습니다: {label_dir}")
    print("label_dir 변수의 경로가 올바른지 확인해주세요.")
else:
    # labels 폴더 안에 있는 모든 .txt 파일 목록을 가져옵니다.
    txt_files = glob(os.path.join(label_dir, "*.txt"))

    renamed_count = 0
    for old_filepath in txt_files:
        # .rf. 가 포함된 파일만 대상으로 합니다.
        if ".rf." in old_filepath:
            dir_path, filename = os.path.split(old_filepath)
            
            base_name_part = filename.split('.rf.')[0]
            clean_name = base_name_part.replace('_jpg', '').replace('_jpeg', '').replace('_png', '')

            new_filename = ""

            try:
                # 규칙 1: 'levle0_'로 시작하는 파일 처리
                if clean_name.startswith("levle0_"):
                    # 예: "levle0_273" -> "levle0_273.txt"
                    number_part = clean_name.split('_')[1]
                    new_filename = f"levle0_{number_part}.txt"
                
                # 규칙 2: 'levle1_', 'levle2_' 등으로 시작하는 파일 처리
                elif clean_name.startswith("levle"):
                    parts = re.split(r'[_]', clean_name)
                    if len(parts) >= 2:
                        level_num = re.sub(r'\D', '', parts[0]) # 'levle1' -> '1'
                        file_num = parts[1]
                        
                        # 새 파일 이름 형식 생성: "1 (273).txt"
                        new_filename = f"{level_num} ({file_num}).txt"
                    else:
                        print(f"[경고] '{filename}' 파일은 예상 형식과 달라 건너뜁니다.")
                        continue
                else:
                    print(f"[경고] '{filename}' 파일은 예상 형식과 달라 건너뜁니다.")
                    continue

                # --- 중복 처리 로직 ---
                new_filepath = os.path.join(dir_path, new_filename)
                
                final_new_filepath = new_filepath
                final_new_filename = new_filename
                counter = 1
                # 만약 변경하려는 파일 이름이 이미 존재한다면
                while os.path.exists(final_new_filepath):
                    # 파일 이름 뒤에 _숫자를 붙여 새로운 이름을 만듭니다.
                    base, ext = os.path.splitext(new_filename)
                    final_new_filename = f"{base}_{counter}{ext}"
                    final_new_filepath = os.path.join(dir_path, final_new_filename)
                    counter += 1
                # --- 중복 처리 로직 끝 ---

                # 최종적으로 결정된 새 파일 이름으로 변경합니다.
                os.rename(old_filepath, final_new_filepath)
                print(f"변경: {filename} -> {final_new_filename}")
                renamed_count += 1

            except Exception as e:
                print(f"[오류] '{filename}' 파일 처리 중 오류 발생: {e}")

    if renamed_count > 0:
        print(f"\n총 {renamed_count}개의 레이블 파일 이름을 성공적으로 변경했습니다.")
    else:
        print("이름을 변경할 파일이 없거나 이미 올바른 형식입니다.")