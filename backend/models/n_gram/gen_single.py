import kenlm
import os
import random
from pyvi import ViTokenizer

class BidirectionalGenerator:
    def __init__(self, model_path, train_data_path, n_gram_order=5):
        print("Đang load mô hình KenLM...")
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}")

        self.model = kenlm.Model(model_path)
        self.n_order = n_gram_order

        print("Đang xây dựng bản đồ từ vựng 2 chiều...")
        self.fwd_map = {}
        self.bwd_map = {}

        with open(train_data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                words = line.split()
                if len(words) < 2: continue

                for i in range(len(words) - 1):
                    for k in range(1, self.n_order):
                        if i + k < len(words):
                            # Map Xuôi
                            fwd_ctx = tuple(words[i : i+k])
                            next_w = words[i+k]
                            if fwd_ctx not in self.fwd_map: self.fwd_map[fwd_ctx] = []
                            self.fwd_map[fwd_ctx].append(next_w)

                            # Map Ngược
                            prev_w = words[i]
                            bwd_ctx = tuple(words[i+1 : i+1+k])
                            if bwd_ctx not in self.bwd_map: self.bwd_map[bwd_ctx] = []
                            self.bwd_map[bwd_ctx].append(prev_w)

    def generate(self, seed_text):
        tokenized = ViTokenizer.tokenize(seed_text).lower()
        current_words = tokenized.split()

        print(f"Input: {current_words}")

        # === 1. CHIỀU NGƯỢC (BACKWARD) ===
        # Chỉ chạy nếu từ đầu tiên hiện tại chưa phải là <s>
        if current_words[0] != "<s>":
            for _ in range(100):
                candidates = None
                search_len = min(len(current_words), self.n_order - 1)

                # Tìm ngữ cảnh
                for k in range(search_len, 0, -1):
                    ctx = tuple(current_words[:k])
                    candidates = self.bwd_map.get(ctx)
                    if candidates: break

                if not candidates: break

                # Chọn từ và chèn vào đầu
                prev_word = random.choice(candidates)
                current_words.insert(0, prev_word)

                # --- KIỂM TRA ĐIỀU KIỆN DỪNG (MỚI) ---
                # Nếu từ vừa sinh ra là <s> thì dừng ngay lập tức
                if prev_word == "<s>":
                    break

        # === 2. CHIỀU XUÔI (FORWARD) ===
        # Chỉ chạy nếu từ cuối cùng hiện tại chưa phải là </s>
        if current_words[-1] != "</s>":
            for _ in range(100):
                candidates = None
                search_len = min(len(current_words), self.n_order - 1)

                # Tìm ngữ cảnh
                for k in range(search_len, 0, -1):
                    ctx = tuple(current_words[-k:])
                    candidates = self.fwd_map.get(ctx)
                    if candidates: break

                if not candidates: break

                # Chọn từ tốt nhất
                best_word = ""
                best_score = -9999
                for word in candidates:
                    sent = " ".join(current_words + [word])
                    score = self.model.score(sent)
                    if score > best_score:
                        best_score = score
                        best_word = word

                # Thêm vào cuối
                current_words.append(best_word)

                # --- KIỂM TRA ĐIỀU KIỆN DỪNG (MỚI) ---
                # Nếu từ vừa sinh ra là </s> thì dừng ngay lập tức
                if best_word == "</s>":
                    break

        # 3. KẾT QUẢ
        # Lọc bỏ <s> và </s> khỏi danh sách trước khi nối chuỗi
        clean_words = [w for w in current_words if w not in ["<s>", "</s>"]]
        return " ".join(clean_words).replace("_", " ")

# --- CHẠY THỬ ---
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data")
    ngram_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(ngram_dir, "checkpoint")
    
    model_file = os.path.join(checkpoint_dir, 'model.bin')
    data_file = os.path.join(data_dir, 'train_data_seg.txt')

    try:
        gen = BidirectionalGenerator(model_file, data_file, n_gram_order=5)

        test_case = "ai kêu là"
        print(f"\n--- TEST: '{test_case}' ---")
        print(f"Kết quả: {gen.generate(test_case)}")

    except Exception as e:
        print(f"Lỗi: {e}")