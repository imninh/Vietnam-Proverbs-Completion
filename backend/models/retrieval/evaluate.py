import pandas as pd
import time
import difflib
import os
import sys
from pathlib import Path
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn"])
    import matplotlib.pyplot as plt
    import seaborn as sns

# ==============================================================================
# 1. TẠO DỮ LIỆU TEST TOÀN DIỆN (TỐI ƯU)
# ==============================================================================
def create_full_stress_test(original_file, max_input_ratio=0.7):

    print(f"🛠  Đang xây dựng ma trận test toàn diện từ {original_file}...")

    if not os.path.exists(original_file):
        print(f"❌ Không tìm thấy: {original_file}")
        return None

    with open(original_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    test_data = []
    append = test_data.append

    for line in tqdm(lines, desc="Processing Data"):
        words = line.lower().split()
        n = len(words)

        if n < 3:
            continue

        ground_truth = " ".join(words)
        max_len = max(1, int(n * max_input_ratio))

        # START (Đầu)
        for k in range(1, max_len + 1):
            append({
                "Loại": "Start (Đầu)",
                "Input_Len": k,
                "Input": " ".join(words[:k]),
                "Ground_Truth": ground_truth
            })

        # END (Cuối)
        for k in range(1, max_len + 1):
            append({
                "Loại": "End (Cuối)",
                "Input_Len": k,
                "Input": " ".join(words[-k:]),
                "Ground_Truth": ground_truth
            })

        # MID (Giữa)
        if n >= 5:
            for k in range(1, max_len + 1):
                for s in range(1, n - k):
                    append({
                        "Loại": "Mid (Giữa)",
                        "Input_Len": k,
                        "Input": " ".join(words[s : s+k]),
                        "Ground_Truth": ground_truth
                    })

    df = pd.DataFrame(test_data)
    print(f"✅ Đã tạo {len(df):,} mẫu test cases từ {len(lines):,} câu gốc.")
    return df


# ==============================================================================
# 2. TỐI ƯU ĐÁNH GIÁ RETRIEVAL
# ==============================================================================
def evaluate_retrieval_full(model, test_df, top_k=5):

    print(f"\n🚀 BẮT ĐẦU ĐÁNH GIÁ TRÊN {len(test_df):,} MẪU...")

    cache_predictions = {}
    cache_similarity = {}
    results = []
    append = results.append
    seq = difflib.SequenceMatcher(None)

    top1_correct = top3_correct = top5_correct = 0
    total_similarity = 0
    start_time = time.time()

    for row in tqdm(test_df.itertuples(), total=len(test_df), desc="Evaluating"):
        inp = row.Input
        truth = row.Ground_Truth

        # Cache predictions
        preds = cache_predictions.get(inp)
        if preds is None:
            preds = model.predict_multiple(inp, top_k=top_k)
            cache_predictions[inp] = preds

        if preds:
            # preds có thể là list của strings hoặc dicts
            if isinstance(preds[0], dict):
                pred_top1 = preds[0]["text"].lower().strip()
                pred_top3 = [p["text"].lower().strip() for p in preds[:3]]
                pred_top5 = [p["text"].lower().strip() for p in preds[:top_k]]
                score = preds[0].get("score", 0.0)
            else:
                # preds là list of strings
                pred_top1 = preds[0].lower().strip()
                pred_top3 = [p.lower().strip() for p in preds[:3]]
                pred_top5 = [p.lower().strip() for p in preds[:top_k]]
                score = 0.0
        else:
            pred_top1 = ""
            pred_top3 = pred_top5 = []
            score = 0.0

        truth_clean = truth

        # Scoring
        top1 = int(pred_top1 == truth_clean)
        top1_correct += top1

        top3 = int(truth_clean in pred_top3)
        top3_correct += top3

        top5 = int(truth_clean in pred_top5)
        top5_correct += top5

        # Similarity (optimized with caching)
        pair_key = (pred_top1, truth_clean)
        sim = cache_similarity.get(pair_key)
        if sim is None:
            seq.set_seqs(pred_top1, truth_clean)
            sim = seq.ratio()
            cache_similarity[pair_key] = sim

        total_similarity += sim

        append({
            "Loại": row.Loại,
            "Input_Len": row.Input_Len,
            "Input": inp,
            "Prediction (Top-1)": pred_top1,
            "Ground Truth": truth_clean,
            "Score": score,
            "Top-1": top1,
            "Top-3": top3,
            "Top-5": top5,
            "Similarity": sim
        })

    duration = time.time() - start_time
    res_df = pd.DataFrame(results)
    N = len(test_df)

    print("\n" + "="*70)
    print(f"📊 BÁO CÁO HIỆU NĂNG RETRIEVAL MODEL - TOÀN DIỆN")
    print("="*70)
    print(f"⏱  Thời gian: {duration:.2f}s ({duration/N*1000:.1f} ms/case)")
    print(f"💾 Cache predict: {len(cache_predictions)} entries")
    print(f"💾 Cache similarity: {len(cache_similarity)} pairs")

    print(f"\n🎯 ĐỘ CHÍNH XÁC:")
    print(f"   • Top-1 Exact Match:     {top1_correct/N*100:>6.2f}%  ({top1_correct:>7,} / {N:,})")
    print(f"   • Top-3 Match:           {top3_correct/N*100:>6.2f}%  ({top3_correct:>7,} / {N:,})")
    print(f"   • Top-5 Match:           {top5_correct/N*100:>6.2f}%  ({top5_correct:>7,} / {N:,})")

    print(f"\n≈  ĐỘ TƯƠNG ĐỒNG:")
    print(f"   • Similarity (Trung bình): {total_similarity/N*100:>6.2f}%")

    # Phân tích theo vị trí
    print("\n" + "-"*70)
    print("1. PHÂN TÍCH THEO VỊ TRÍ (Start/Mid/End):")
    group_type = res_df.groupby("Loại")[["Top-1", "Top-3", "Top-5", "Similarity"]].mean() * 100
    print(group_type.round(2))

    # Phân tích theo độ dài input
    print("\n" + "-"*70)
    print("2. PHÂN TÍCH THEO ĐỘ DÀI INPUT (10 độ dài đầu):")
    group_len = res_df.groupby("Input_Len")[["Top-1", "Top-3", "Top-5", "Similarity"]].mean().head(10) * 100
    print(group_len.round(2))
    print("="*70)

    return res_df


# ==============================================================================
# 3. VẼ ĐỒ THỊ
# ==============================================================================
def plot_retrieval_results(res_df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['font.size'] = 10

    # --- ĐỒ THỊ CHÍNH ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Bar chart
    type_stats = res_df.groupby("Loại")[["Top-1", "Top-3", "Top-5", "Similarity"]].mean() * 100
    type_stats.plot(kind='bar', ax=axes[0, 0], color=['#e74c3c', '#f39c12', '#2ecc71', '#3498db'])
    axes[0, 0].set_title('Hiệu năng theo vị trí (Start/Mid/End)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Phần trăm (%)')
    axes[0, 0].set_xlabel('Vị trí')
    axes[0, 0].legend(['Top-1', 'Top-3', 'Top-5', 'Similarity'], fontsize=9)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 105)

    # Plot 2: Line chart
    len_stats = res_df.groupby("Input_Len")[["Top-1", "Top-3", "Top-5", "Similarity"]].mean() * 100
    axes[0, 1].plot(len_stats.index, len_stats["Top-1"], marker='o', label='Top-1', color='#e74c3c', linewidth=2)
    axes[0, 1].plot(len_stats.index, len_stats["Top-3"], marker='s', label='Top-3', color='#f39c12', linewidth=2)
    axes[0, 1].plot(len_stats.index, len_stats["Top-5"], marker='^', label='Top-5', color='#2ecc71', linewidth=2)
    axes[0, 1].plot(len_stats.index, len_stats["Similarity"], marker='d', label='Similarity', color='#3498db', linewidth=2)
    axes[0, 1].set_title('Hiệu năng theo độ dài input', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Độ dài Input (từ)')
    axes[0, 1].set_ylabel('Phần trăm (%)')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Pie chart
    correct_dist = res_df["Top-1"].value_counts()
    labels = ['Sai', 'Đúng'] if len(correct_dist) > 1 else ['Đúng']
    colors = ['#e74c3c', '#2ecc71']
    axes[1, 0].pie(correct_dist.values, 
                   labels=[f'{l}\n{v} ({v/len(res_df)*100:.1f}%)' for l, v in zip(labels, correct_dist.values)], 
                   autopct='', colors=colors[:len(correct_dist)], startangle=90)
    axes[1, 0].set_title('Phân phối Top-1 Đúng/Sai', fontsize=12, fontweight='bold')

    # Plot 4: Scatter
    scatter = axes[1, 1].scatter(res_df["Input_Len"], res_df["Similarity"] * 100,
                                 c=res_df["Top-1"], cmap='RdYlGn', s=30, alpha=0.6, 
                                 edgecolors='black', linewidth=0.5)
    axes[1, 1].set_title('Input Length vs Similarity', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Độ dài Input (từ)')
    axes[1, 1].set_ylabel('Similarity (%)')
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Đúng/Sai')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'retrieval_evaluation_report.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] {output_file}")
    plt.close()

    # --- ĐỒ THỊ BỔ SUNG ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, type_name in enumerate(res_df["Loại"].unique()):
        type_data = res_df[res_df["Loại"] == type_name]
        len_stats_type = type_data.groupby("Input_Len")[["Top-1", "Similarity"]].mean() * 100

        axes[idx].plot(len_stats_type.index, len_stats_type["Top-1"],
                      marker='o', linewidth=2, markersize=6, color='#e74c3c', label='Top-1')
        axes[idx].plot(len_stats_type.index, len_stats_type["Similarity"],
                      marker='s', linewidth=2, markersize=6, color='#3498db', label='Similarity')
        axes[idx].set_title(f'{type_name}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Độ dài Input (từ)')
        axes[idx].set_ylabel('Phần trăm (%)')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(0, 105)
        axes[idx].legend(fontsize=9)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'retrieval_evaluation_by_type.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_file}")
    plt.close()

    print(f"\n✅ Tất cả đồ thị đã được lưu vào: {output_dir}")


# ==============================================================================
# 4. MAIN
# ==============================================================================
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    retrieval_dir = Path(__file__).parent
    checkpoint_dir = retrieval_dir / "checkpoint"
    eval_dir = project_root / "evaluation" / "retrieval"
    
    model_file = checkpoint_dir / "retrieval.pkl"
    raw_dataset_file = data_dir / "dataset.txt"

    try:
        sys.path.insert(0, str(retrieval_dir))
        from train import RetrievalModel

        print("\n" + "="*70)
        print("🔍 ĐÁNH GIÁ TOÀN DIỆN - RETRIEVAL MODEL")
        print("="*70)

        if not model_file.exists():
            print(f"❌ Model không tìm thấy: {model_file}")
            sys.exit(1)

        print(f"\n📂 Loading model từ: {model_file}")
        model = RetrievalModel()
        model.load(str(model_file))
        print("✓ Model loaded successfully")

        print(f"\n📂 Loading training data từ: {raw_dataset_file}")
        full_test_df = create_full_stress_test(str(raw_dataset_file), max_input_ratio=0.7)
        
        if full_test_df is None:
            sys.exit(1)
        
        print(f"📊 Tổng test cases: {len(full_test_df):,}")

        # Đánh giá
        df_final = evaluate_retrieval_full(model, full_test_df, top_k=5)

        # Vẽ đồ thị
        eval_dir.mkdir(parents=True, exist_ok=True)
        plot_retrieval_results(df_final, str(eval_dir))

        # Lưu CSV
        results_file = eval_dir / "evaluation_results.csv"
        df_final.to_csv(results_file, index=False, encoding='utf-8')
        print(f"✓ Results saved to: {results_file}")

        # Top 10 cases sai
        print("\n" + "="*70)
        print("🔍 TOP 10 TRƯỜNG HỢP SAI (Similarity thấp nhất):")
        print("="*70)

        wrong_cases = df_final[df_final["Top-1"] == 0].copy()

        if not wrong_cases.empty:
            wrong_cases = wrong_cases.sort_values("Similarity")
            pd.set_option('display.max_colwidth', 60)
            print("\n")
            print(wrong_cases[['Loại', 'Input', 'Prediction (Top-1)', 'Ground Truth', 'Similarity']].head(10).to_string())
        else:
            print("\n✅ Xuất sắc! Model đúng 100%.")

        print("\n" + "="*70)
        print("✅ ĐÁNH GIÁ HOÀN TẤT!")
        print("="*70)

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
