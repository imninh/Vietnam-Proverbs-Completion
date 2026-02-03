import pandas as pd
import random
import time
import difflib
import os
import sys
from pathlib import Path
from pyvi import ViTokenizer
from bm25_scorer import calculate_bm25

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "n_gram"))
from gen_mul_layer import BidirectionalBeamGenerator

import matplotlib.pyplot as plt
import seaborn as sns

# Cài đặt tqdm để theo dõi tiến độ (vì chạy full data sẽ lâu)
try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

# ==============================================================================
# 1. HÀM ENSEMBLE GENERATION (N-gram + BM25)
# ==============================================================================
class EnsembleGenerator:
    """Wrapper cho Ensemble generation"""
    
    def __init__(self, model_file, data_file, n_gram_order=5, num_candidates=5):
        self.generator = BidirectionalBeamGenerator(model_file, data_file, n_gram_order=n_gram_order)
        self.num_candidates = num_candidates
    
    def generate_best_cases(self, input_text, num_results=1):
        """
        Generate candidates và rank theo BM25
        
        Args:
            input_text (str): Text input
            num_results (int): Số kết quả trả về
        
        Returns:
            list: List of (score, text) tuples
        """
        try:
            # Bước 1: Generate candidates từ N-gram
            ngram_results = self.generator.generate_best_cases(input_text, num_results=self.num_candidates)
            
            if not ngram_results:
                return []
            
            # Bước 2: Extract candidates
            candidates = [text for _, text in ngram_results]
            
            # Bước 3: Score với BM25
            scored_results = []
            for candidate in candidates:
                bm25_score = calculate_bm25(input_text, candidate)
                scored_results.append((bm25_score, candidate))
            
            # Bước 4: Sort by BM25 score (descending)
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # Trả về top num_results
            return scored_results[:num_results]
        
        except Exception as e:
            print(f"⚠️ Error in ensemble generation: {e}")
            return []

# ==============================================================================
# 2. HÀM TẠO DỮ LIỆU TEST TOÀN DIỆN (FULL STRESS TEST)
# ==============================================================================
def create_full_stress_test(original_file, max_input_ratio=0.7):
    """
    Tạo ma trận test phủ kín toàn bộ dataset.
    Không lấy mẫu ngẫu nhiên mà vét cạn mọi trường hợp.
    """
    print(f"🛠 Đang xây dựng ma trận test toàn diện từ {original_file}...")
    test_data = []

    # Tạo file mẫu nếu chưa có
    if not os.path.exists(original_file):
        sample_text = """
        Công cha như núi Thái Sơn
        Nghĩa mẹ như nước trong nguồn chảy ra.
        Anh em như thể tay chân
        Rách lành đùm bọc dở hay đỡ đần.
        """
        with open(original_file, "w", encoding="utf-8") as f:
            f.write(sample_text.strip())

    with open(original_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Duyệt qua từng dòng với thanh tiến trình
    for line in tqdm(lines, desc="Processing Data"):
        tokenized_line = ViTokenizer.tokenize(line).lower()
        words = tokenized_line.split()
        n = len(words)

        # Bỏ qua câu quá ngắn
        if n < 3: continue

        ground_truth = " ".join(words)
        max_test_len = max(1, int(n * max_input_ratio))

        # -------------------------------------------------
        # 1. CASE START (Đầu câu -> Test Gen Xuôi)
        # -------------------------------------------------
        for k in range(1, max_test_len + 1):
            test_data.append({
                "Loại": "Start (Đầu)",
                "Input Len": k,
                "Input": " ".join(words[:k]),
                "Ground_Truth": ground_truth
            })

        # -------------------------------------------------
        # 2. CASE END (Cuối câu -> Test Gen Ngược)
        # -------------------------------------------------
        for k in range(1, max_test_len + 1):
            test_data.append({
                "Loại": "End (Cuối)",
                "Input Len": k,
                "Input": " ".join(words[-k:]),
                "Ground_Truth": ground_truth
            })

        # -------------------------------------------------
        # 3. CASE MID (Giữa câu -> Test Gen 2 Chiều)
        # -------------------------------------------------
        if n >= 5:
            # Quét độ dài input từ 1 đến max
            for k in range(1, max_test_len + 1):
                # Trượt cửa sổ
                for start_idx in range(1, n - k):
                    test_data.append({
                        "Loại": "Mid (Giữa)",
                        "Input Len": k,
                        "Input": " ".join(words[start_idx : start_idx + k]),
                        "Ground_Truth": ground_truth
                    })

    df = pd.DataFrame(test_data)
    print(f"✅ Đã tạo {len(df)} mẫu test cases từ {len(lines)} câu gốc.")
    return df

# ==============================================================================
# 3. HÀM ĐÁNH GIÁ CHI TIẾT
# ==============================================================================
def evaluate_full_dataset(generator, test_df):
    print(f"\n🚀 BẮT ĐẦU ĐÁNH GIÁ TRÊN {len(test_df)} MẪU...")

    results = []
    correct_count = 0
    total_similarity = 0
    start_time = time.time()

    # Chạy vòng lặp đánh giá
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        inp = row['Input']
        truth = row['Ground_Truth']

        try:
            # --- GỌI MODEL ENSEMBLE (N-gram + BM25) ---
            gen_output = generator.generate_best_cases(inp, num_results=1)

            if not gen_output:
                pred_clean = ""
                score = -999.0
            else:
                # Unpack kết quả: (bm25_score, text)
                score, formatted_text = gen_output[0]

                # --- CHUẨN HÓA KẾT QUẢ ĐỂ SO SÁNH ---
                # Model trả về dạng thơ đẹp -> Cần đưa về dạng tokenized thường
                pred_clean = formatted_text.replace("\n", " ").lower()
                pred_clean = pred_clean.replace(".", "").replace(",", "").strip()
                # Quan trọng: Phải tokenize lại thì mới khớp với Ground Truth (vd: thái_sơn)
                pred_clean = ViTokenizer.tokenize(pred_clean)

        except Exception as e:
            pred_clean = f"Error: {e}"
            score = -999.0

        # Chuẩn hóa Ground Truth
        truth_clean = truth.replace(".", "").strip()

        # --- CHẤM ĐIỂM ---
        # 1. Exact Match
        is_exact = 1 if pred_clean == truth_clean else 0
        correct_count += is_exact

        # 2. Similarity
        sim = difflib.SequenceMatcher(None, pred_clean, truth_clean).ratio()
        total_similarity += sim

        results.append({
            "Loại": row['Loại'],
            "Input Len": row['Input Len'],
            "Input": inp,
            "Kết quả Gen": pred_clean,
            "Đáp án Gốc": truth_clean,
            "Điểm Model": score,
            "Đúng": is_exact,
            "Độ giống": sim
        })

    total_time = time.time() - start_time

    # --- TẠO BÁO CÁO ---
    res_df = pd.DataFrame(results)
    if len(test_df) > 0:
        acc = correct_count / len(test_df) * 100
        avg_sim = total_similarity / len(test_df) * 100
    else:
        acc = 0; avg_sim = 0

    print("\n" + "="*60)
    print(f"📊 BÁO CÁO HIỆU NĂNG TOÀN DIỆN (ENSEMBLE)")
    print("="*60)
    print(f"⏱  Thời gian: {total_time:.2f}s ({total_time/len(test_df)*1000:.1f} ms/câu)")
    print(f"🎯 Độ chính xác tuyệt đối (Exact Match): {acc:.2f}%")
    print(f"≈  Độ tương đồng trung bình (Similarity):  {avg_sim:.2f}%")

    # --- PHÂN TÍCH 1: THEO VỊ TRÍ ---
    print("-" * 60)
    print("1. PHÂN TÍCH THEO VỊ TRÍ (Start/Mid/End):")
    # Xem model giỏi chiều nào hơn
    group_type = res_df.groupby("Loại")[["Đúng", "Độ giống"]].mean() * 100
    print(group_type.round(2))

    # --- PHÂN TÍCH 2: THEO ĐỘ DÀI INPUT ---
    print("-" * 60)
    print("2. PHÂN TÍCH THEO ĐỘ DÀI INPUT (10 độ dài đầu):")
    # Input càng ngắn càng khó đoán
    group_len = res_df.groupby("Input Len")[["Đúng", "Độ giống"]].mean().head(10) * 100
    print(group_len.round(2))
    print("="*60)

    return res_df

# ==============================================================================
# 4. HÀM VẼ ĐỒ THỊ
# ==============================================================================
def plot_evaluation_results(res_df, output_dir):
    """
    Vẽ 4 đồ thị phân tích chi tiết kết quả đánh giá
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Cấu hình seaborn
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['font.size'] = 10
    
    # --- ĐỒ THỊ 1: Hiệu năng theo vị trí (Start/Mid/End) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Bar chart - Accuracy by Type
    type_stats = res_df.groupby("Loại")[["Đúng", "Độ giống"]].mean() * 100
    type_stats.plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#3498db'])
    axes[0, 0].set_title('Hiệu năng theo vị trí (Start/Mid/End) - ENSEMBLE', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Phần trăm (%)')
    axes[0, 0].set_xlabel('Vị trí')
    axes[0, 0].legend(['Chính xác', 'Độ giống'])
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Line chart - Performance by Input Length
    len_stats = res_df.groupby("Input Len")[["Đúng", "Độ giống"]].mean() * 100
    axes[0, 1].plot(len_stats.index, len_stats["Đúng"], marker='o', label='Chính xác', color='#2ecc71', linewidth=2)
    axes[0, 1].plot(len_stats.index, len_stats["Độ giống"], marker='s', label='Độ giống', color='#3498db', linewidth=2)
    axes[0, 1].set_title('Hiệu năng theo độ dài input - ENSEMBLE', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Độ dài Input (từ)')
    axes[0, 1].set_ylabel('Phần trăm (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Pie chart - Distribution of Correct/Wrong
    correct_dist = res_df["Đúng"].value_counts()
    labels = ['Sai', 'Đúng'] if len(correct_dist) > 1 else ['Đúng']
    colors = ['#e74c3c', '#2ecc71']
    axes[1, 0].pie(correct_dist.values, labels=[f'{l}\n{v} ({v/len(res_df)*100:.1f}%)' for l, v in zip(labels, correct_dist.values)], 
                   autopct='', colors=colors[:len(correct_dist)], startangle=90)
    axes[1, 0].set_title('Phân phối Đúng/Sai - ENSEMBLE', fontsize=12, fontweight='bold')
    
    # Plot 4: Scatter - Input Length vs Similarity
    scatter = axes[1, 1].scatter(res_df["Input Len"], res_df["Độ giống"] * 100, 
                                 c=res_df["Đúng"], cmap='RdYlGn', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1, 1].set_title('Input Length vs Độ Giống - ENSEMBLE', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Độ dài Input (từ)')
    axes[1, 1].set_ylabel('Độ giống (%)')
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Đúng/Sai')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Lưu file
    output_file = os.path.join(output_dir, 'evaluation_ensemble_report.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Đồ thị lưu tại: {output_file}")
    plt.close()
    
    # --- ĐỒ THỊ BỔ SUNG: Chi tiết theo từng type ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, type_name in enumerate(res_df["Loại"].unique()):
        type_data = res_df[res_df["Loại"] == type_name]
        len_stats_type = type_data.groupby("Input Len")[["Độ giống"]].mean() * 100
        
        axes[idx].plot(len_stats_type.index, len_stats_type["Độ giống"], 
                      marker='o', linewidth=2, markersize=6, color='#3498db')
        axes[idx].set_title(f'{type_name}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Độ dài Input (từ)')
        axes[idx].set_ylabel('Độ giống (%)')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(0, 105)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'evaluation_ensemble_by_type.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Đồ thị chi tiết lưu tại: {output_file}")
    plt.close()
    
    print(f"✅ Tất cả đồ thị đã được lưu vào: {output_dir}")
    return res_df

# ==============================================================================
# 5. CHẠY THỰC TẾ
# ==============================================================================
if __name__ == "__main__":
    # Cấu hình đường dẫn
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    ensemble_dir = Path(__file__).parent
    ngram_checkpoint_dir = project_root / "models" / "n_gram" / "checkpoint"
    eval_dir = project_root / "evaluation" / "ensemble"
    
    model_file = str(ngram_checkpoint_dir / 'model.bin')
    data_file = str(data_dir / 'train_data_seg.txt')
    raw_dataset = str(data_dir / "dataset.txt")

    try:
        # Kiểm tra file
        if not os.path.exists(model_file):
            print(f"❌ Lỗi: Không tìm thấy model.bin tại {model_file}")
            exit(1)
        
        if not os.path.exists(data_file):
            print(f"❌ Lỗi: Không tìm thấy train_data_seg.txt tại {data_file}")
            exit(1)

        print("⚠️ Đang khởi tạo Ensemble Generator...")
        ensemble_gen = EnsembleGenerator(model_file, data_file, n_gram_order=5, num_candidates=5)

        # 1. Tạo dataset test toàn diện
        full_test_df = create_full_stress_test(raw_dataset, max_input_ratio=0.7)
        
        # Lưu file test data
        test_data_file = data_dir / 'dataset_ensemble_test.csv'
        full_test_df.to_csv(test_data_file, index=False, encoding='utf-8')
        print(f"✅ Test data đã lưu tại: {test_data_file}")

        # 2. Chạy đánh giá
        df_final = evaluate_full_dataset(ensemble_gen, full_test_df)

        # 3. Vẽ đồ thị và lưu
        plot_evaluation_results(df_final, str(eval_dir))

        # 4. Xuất các câu sai để debug
        print("\n🔍 TOP 5 CÂU SAI ĐIỂN HÌNH:")
        wrong_cases = df_final[df_final["Đúng"] == 0].head(5)
        if not wrong_cases.empty:
            pd.set_option('display.max_colwidth', None)
            print(wrong_cases[['Loại', 'Input', 'Kết quả Gen', 'Đáp án Gốc', 'Độ giống']])
        else:
            print("✅ Xuất sắc! Model đúng 100%.")

    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()
