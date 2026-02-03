import kenlm
import os
import random
from pyvi import ViTokenizer

class BidirectionalMultiGenerator:
    def __init__(self, model_path, train_data_path, n_gram_order=5):
        print("[Multi-Gen] Đang load mô hình KenLM...")
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}")

        self.model = kenlm.Model(model_path)
        self.n_order = n_gram_order

        print("[Multi-Gen] Đang xây dựng bản đồ từ vựng...")
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

    # --- HÀM HỖ TRỢ: ĐỊNH DẠNG THƠ (MỚI) ---
    def format_poetry(self, raw_text):
        """
        Chuyển chuỗi dài thành định dạng thơ:
        - Ngắt dòng khi gặp dấu chấm.
        - Viết hoa chữ cái đầu câu.
        """
        # 1. Thay thế gạch dưới
        text = raw_text.replace("_", " ")

        # 2. Tách thành các vế dựa trên dấu chấm
        # Tokenizer thường tách dấu chấm ra (ví dụ "thái sơn .") nên ta split theo dấu chấm
        verses = text.split(".")

        formatted_verses = []
        for verse in verses:
            verse = verse.strip()
            if verse:
                # Viết hoa chữ cái đầu tiên
                formatted_verse = verse[0].upper() + verse[1:]
                formatted_verses.append(formatted_verse)

        # 3. Nối lại bằng xuống dòng
        return "\n".join(formatted_verses)

    # --- HÀM SINH 1 CÂU ---
    def generate_one_sample(self, seed_words, top_k=5):
        current_words = list(seed_words)

        # 1. CHIỀU NGƯỢC
        if current_words[0] != "<s>":
            for _ in range(50):
                candidates = None
                search_len = min(len(current_words), self.n_order - 1)
                for k in range(search_len, 0, -1):
                    ctx = tuple(current_words[:k])
                    candidates = self.bwd_map.get(ctx)
                    if candidates: break

                if not candidates: break
                prev_word = random.choice(candidates)
                current_words.insert(0, prev_word)
                if prev_word == "<s>": break

        # 2. CHIỀU XUÔI
        if current_words[-1] != "</s>":
            for _ in range(50):
                candidates = None
                search_len = min(len(current_words), self.n_order - 1)
                for k in range(search_len, 0, -1):
                    ctx = tuple(current_words[-k:])
                    candidates = self.fwd_map.get(ctx)
                    if candidates: break

                if not candidates: break

                # Top-K Sampling
                unique_candidates = list(set(candidates))
                scored_candidates = []
                for word in unique_candidates:
                    sent = " ".join(current_words + [word])
                    score = self.model.score(sent)
                    scored_candidates.append((score, word))

                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                actual_k = min(len(scored_candidates), top_k)
                top_choices = scored_candidates[:actual_k]

                best_word = random.choice(top_choices)[1]
                current_words.append(best_word)
                if best_word == "</s>": break

        return current_words

    # --- HÀM SINH BATCH ---
    def generate_batch(self, seed_text, num_sentences=10, top_k=5):
        tokenized = ViTokenizer.tokenize(seed_text).lower()
        seed_words = tokenized.split()

        print(f"Input: {seed_words}")
        print(f"Đang sinh {num_sentences} kết quả (Top-{top_k})...")

        unique_results = set()
        final_list = []
        max_trials = num_sentences * 5

        for _ in range(max_trials):
            words = self.generate_one_sample(seed_words, top_k)

            # Lấy chuỗi raw để lọc trùng và tính điểm
            clean_words = [w for w in words if w not in ["<s>", "</s>"]]
            raw_text = " ".join(clean_words) # Vẫn giữ gạch dưới để tính điểm chính xác

            if raw_text not in unique_results:
                unique_results.add(raw_text)
                score = self.model.score(raw_text)

                # Gọi hàm format để hiển thị đẹp
                pretty_text = self.format_poetry(raw_text)

                final_list.append((score, pretty_text))

            if len(final_list) >= num_sentences:
                break

        final_list.sort(key=lambda x: x[0], reverse=True)
        return final_list

# --- CHẠY THỬ ---
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data")
    ngram_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(ngram_dir, "checkpoint")
    
    model_file = os.path.join(checkpoint_dir, 'model.bin')
    data_file = os.path.join(data_dir, 'train_data_seg.txt')

    try:
        multi_gen = BidirectionalMultiGenerator(model_file, data_file, n_gram_order=5)

        # Nhập từ khóa
        test_input = "người bạn cũ bây giờ"

        # Sinh kết quả
        results = multi_gen.generate_batch(test_input, num_sentences=3, top_k=5)

        print(f"\nKẾT QUẢ CHO TỪ KHÓA: '{test_input}'")
        print("=" * 60)

        if not results:
            print("Không tìm thấy kết quả phù hợp.")
        else:
            for i, (score, text) in enumerate(results, 1):
                print(f"#{i} (Điểm: {score:.2f})")
                print(text) # In ra bài thơ đã format
                print("-" * 30) # Đường kẻ phân cách giữa các bài

    except Exception as e:
        print(f"Lỗi: {e}")