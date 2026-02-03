import kenlm
import os
import heapq
from pyvi import ViTokenizer

class BidirectionalBeamGenerator:
    def __init__(self, model_path, train_data_path, n_gram_order=5):
        print("[Beam Search] Đang load mô hình KenLM...")
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}")

        self.model = kenlm.Model(model_path)
        self.n_order = n_gram_order

        print("[Beam Search] Đang xây dựng bản đồ từ vựng...")
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
                            if fwd_ctx not in self.fwd_map: self.fwd_map[fwd_ctx] = set()
                            self.fwd_map[fwd_ctx].add(next_w)

                            # Map Ngược
                            prev_w = words[i]
                            bwd_ctx = tuple(words[i+1 : i+1+k])
                            if bwd_ctx not in self.bwd_map: self.bwd_map[bwd_ctx] = set()
                            self.bwd_map[bwd_ctx].add(prev_w)

    def expand_backward(self, seed_words, beam_width=10):
        current_beams = [list(seed_words)]
        completed_prefixes = []

        for _ in range(50):
            next_beams = []
            for words in current_beams:
                if words[0] == "<s>":
                    completed_prefixes.append(words)
                    continue

                candidates = None
                search_len = min(len(words), self.n_order - 1)
                for k in range(search_len, 0, -1):
                    ctx = tuple(words[:k])
                    if ctx in self.bwd_map:
                        candidates = self.bwd_map[ctx]
                        break

                if candidates:
                    for prev_w in candidates:
                        new_sent = [prev_w] + words
                        next_beams.append(new_sent)

            if not next_beams: break
            if len(next_beams) > beam_width * 5:
                next_beams = next_beams[:beam_width * 5]
            current_beams = next_beams
            if not current_beams: break

        return completed_prefixes if completed_prefixes else [list(seed_words)]

    def expand_forward_beam(self, prefixes, beam_width=5):
        current_beams = []
        for words in prefixes:
            score = self.model.score(" ".join(words))
            current_beams.append((score, words))

        current_beams.sort(key=lambda x: x[0], reverse=True)
        current_beams = current_beams[:beam_width]

        final_results = []

        for _ in range(50):
            next_beams = []
            all_ended = True

            for score, words in current_beams:
                if words[-1] == "</s>":
                    final_results.append((score, words))
                    continue

                all_ended = False
                candidates = None
                search_len = min(len(words), self.n_order - 1)
                for k in range(search_len, 0, -1):
                    ctx = tuple(words[-k:])
                    if ctx in self.fwd_map:
                        candidates = self.fwd_map[ctx]
                        break

                if candidates:
                    for next_w in candidates:
                        new_words = words + [next_w]
                        new_sent_str = " ".join(new_words)
                        new_score = self.model.score(new_sent_str)
                        next_beams.append((new_score, new_words))

            if all_ended: break
            if not next_beams: break

            next_beams.sort(key=lambda x: x[0], reverse=True)
            current_beams = next_beams[:beam_width]

        for score, words in current_beams:
            final_results.append((score, words))

        unique_map = {}
        for score, words in final_results:
            text = " ".join(words)
            if text not in unique_map:
                unique_map[text] = score
            else:
                if score > unique_map[text]:
                    unique_map[text] = score

        # Trả về danh sách tuples (Text, Score)
        sorted_results = sorted(unique_map.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:beam_width]

    def format_poetry(self, raw_text):
        # Hàm này nhận vào string, nếu nhận float sẽ lỗi
        clean_text = raw_text.replace("<s>", "").replace("</s>", "").strip()
        text = clean_text.replace("_", " ")
        verses = text.split(".")
        formatted_verses = []
        for verse in verses:
            verse = verse.strip()
            if verse:
                formatted_verse = verse[0].upper() + verse[1:]
                formatted_verses.append(formatted_verse)
        return "\n".join(formatted_verses)

    def generate_best_cases(self, seed_text, num_results=5):
        tokenized = ViTokenizer.tokenize(seed_text).lower()
        seed_words = tokenized.split()

        print(f"Input: {seed_words}")
        print(f"Đang tìm {num_results} kết quả tốt nhất theo xác suất...")

        prefixes = self.expand_backward(seed_words, beam_width=num_results*2)
        results = self.expand_forward_beam(prefixes, beam_width=num_results)

        formatted_results = []

        # --- SỬA LỖI TẠI ĐÂY ---
        # Kết quả trả về là (text, score) chứ không phải (score, text)
        for text, score in results:
            pretty_text = self.format_poetry(text)
            formatted_results.append((score, pretty_text))

        return formatted_results

# --- CHẠY THỬ ---
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data")
    ngram_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(ngram_dir, "checkpoint")
    
    model_file = os.path.join(checkpoint_dir, 'model.bin')
    data_file = os.path.join(data_dir, 'train_data_seg.txt')

    try:
        beam_gen = BidirectionalBeamGenerator(model_file, data_file, n_gram_order=5)

        test_input = "người bạn cũ bây giờ"

        results = beam_gen.generate_best_cases(test_input, num_results=5)

        print(f"\nTOP KẾT QUẢ CHO: '{test_input}'")
        print("=" * 60)

        if not results:
            print("Không tìm thấy kết quả phù hợp.")
        else:
            for i, (score, text) in enumerate(results, 1):
                print(f"Hạng #{i} (Score: {score:.2f})")
                print(text)
                print("-" * 30)

    except Exception as e:
        print(f"Lỗi: {e}")