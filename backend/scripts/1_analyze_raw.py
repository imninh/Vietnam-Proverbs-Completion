import re
import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path để import được từ backend
sys.path.append(str(Path(__file__).parent.parent))


def analyze_dataset(file_path):
    """Phân tích chi tiết dataset gốc"""
    
    print("📊 PHÂN TÍCH DATASET GỐC")
    
    # Đọc file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f" Không tìm thấy file: {file_path}")
        print(f" Đường dẫn hiện tại: {Path.cwd()}")
        print(f" Hãy đảm bảo file dataset nằm ở: backend/data/raw/cadao_raw.txt")
        return False
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return False
    
    print(f"\n📁 File: {file_path}")
    print(f"✓ Đọc thành công!")
    
    # ========== 1. THỐNG KÊ CƠ BẢN ==========
    print(f"\n{'─'*70}")
    print("1️⃣  THỐNG KÊ CƠ BẢN")
    print(f"{'─'*70}")
    
    total_lines = len(lines)
    empty_lines = sum(1 for line in lines if not line.strip())
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    
    print(f"   Tổng số dòng:        {total_lines:,}")
    print(f"   Dòng rỗng:           {empty_lines:,}")
    print(f"   Dòng có nội dung:    {len(non_empty_lines):,}")
    print(f"   Tỷ lệ dòng rỗng:     {empty_lines/total_lines*100:.1f}%")
    
    # ========== 2. ĐỘ DÀI CÂU ==========
    print(f"\n{'─'*70}")
    print("2️⃣  ĐỘ DÀI CÂU")
    print(f"{'─'*70}")
    
    if non_empty_lines:
        lengths = [len(line.split()) for line in non_empty_lines]
        char_lengths = [len(line) for line in non_empty_lines]
        
        print(f"   Độ dài theo từ:")
        print(f"      Trung bình:  {sum(lengths)/len(lengths):.1f} từ")
        print(f"      Ngắn nhất:   {min(lengths)} từ")
        print(f"      Dài nhất:    {max(lengths)} từ")
        
        print(f"\n   Độ dài theo ký tự:")
        print(f"      Trung bình:  {sum(char_lengths)/len(char_lengths):.1f} ký tự")
        print(f"      Ngắn nhất:   {min(char_lengths)} ký tự")
        print(f"      Dài nhất:    {max(char_lengths)} ký tự")
        
        # Phân bố độ dài
        print(f"\n   Phân bố độ dài (theo từ):")
        length_distribution = Counter(lengths)
        for length in sorted(length_distribution.keys())[:10]:
            count = length_distribution[length]
            bar = '█' * int(count / len(non_empty_lines) * 50)
            print(f"      {length:2d} từ: {count:3d} câu {bar}")
    
    # ========== 3. KÝ TỰ ĐẶC BIỆT ==========
    print(f"\n{'─'*70}")
    print("3️⃣  KÝ TỰ ĐẶC BIỆT & SỐ")
    print(f"{'─'*70}")
    
    special_chars = set()
    has_numbers = []
    
    for line in lines:
        special_chars.update(re.findall(r'[^\w\s]', line))
        if re.search(r'\d', line):
            has_numbers.append(line.strip())
    
    print(f"   Ký tự đặc biệt tìm thấy: {sorted(special_chars)}")
    print(f"   Số dòng có chứa số:      {len(has_numbers)}")
    
    if has_numbers:
        print(f"\n   📝 5 ví dụ dòng có số:")
        for i, line in enumerate(has_numbers[:5], 1):
            print(f"      {i}. {line[:60]}{'...' if len(line) > 60 else ''}")
    
    # ========== 4. CÂU TRÙNG LẶP ==========
    print(f"\n{'─'*70}")
    print("4️⃣  CÂU TRÙNG LẶP")
    print(f"{'─'*70}")
    
    line_counts = Counter(line.strip().lower() for line in lines if line.strip())
    duplicates = {line: count for line, count in line_counts.items() if count > 1}
    
    print(f"   Số câu duy nhất:     {len(line_counts):,}")
    print(f"   Số câu bị trùng:     {len(duplicates):,}")
    print(f"   Tỷ lệ trùng lặp:     {len(duplicates)/len(line_counts)*100:.1f}%")
    
    if duplicates:
        print(f"\n   📝 Top 5 câu trùng nhiều nhất:")
        for i, (line, count) in enumerate(sorted(duplicates.items(), 
                                                  key=lambda x: x[1], 
                                                  reverse=True)[:5], 1):
            preview = line[:50] + '...' if len(line) > 50 else line
            print(f"      {i}. '{preview}' - {count} lần")
    
    # ========== 5. MẪU DỮ LIỆU ==========
    print(f"\n{'─'*70}")
    print("5️⃣  MẪU DỮ LIỆU (10 câu đầu)")
    print(f"{'─'*70}")
    
    for i, line in enumerate(non_empty_lines[:10], 1):
        # Highlight vấn đề
        issues = []
        if re.search(r'^\d+[\.\):]', line):
            issues.append("🔢 Có số đầu dòng")
        if len(line.split()) < 4:
            issues.append("⚠️  Quá ngắn")
        if re.search(r'[^\w\s,\.]', line):
            issues.append("❗ Có ký tự lạ")
        
        issue_str = f" [{', '.join(issues)}]" if issues else ""
        print(f"   {i:2d}. {line[:60]}{'...' if len(line) > 60 else ''}{issue_str}")
    
    # ========== 6. VẤN ĐỀ CẦN XỬ LÝ ==========
    print(f"\n{'─'*70}")
    print("6️⃣  VẤN ĐỀ CẦN XỬ LÝ")
    print(f"{'─'*70}")
    
    issues_found = []
    
    if empty_lines > 0:
        issues_found.append(f"✓ Xóa {empty_lines} dòng rỗng")
    
    if duplicates:
        issues_found.append(f"✓ Xóa {len(duplicates)} câu trùng lặp")
    
    if has_numbers:
        issues_found.append(f"✓ Xử lý {len(has_numbers)} dòng có số")
    
    if special_chars:
        issues_found.append(f"✓ Xử lý {len(special_chars)} loại ký tự đặc biệt")
    
    short_lines = sum(1 for line in non_empty_lines if len(line.split()) < 4)
    if short_lines > 0:
        issues_found.append(f"✓ Xóa {short_lines} câu quá ngắn (< 4 từ)")
    
    if issues_found:
        print("\n   Cần thực hiện:")
        for issue in issues_found:
            print(f"      {issue}")
    else:
        print("\n   ✅ Dataset khá sạch, ít vấn đề!")
    
    # ========== DỰ ĐOÁN SAU KHI LÀM SẠCH ==========
    print(f"\n{'─'*70}")
    print("7️⃣  DỰ ĐOÁN SAU KHI LÀM SẠCH")
    print(f"{'─'*70}")
    
    estimated_clean = len(line_counts) - len(duplicates)
    estimated_clean -= short_lines
    estimated_clean -= sum(1 for line in non_empty_lines if not re.search(r'[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', line.lower()))
    
    retention_rate = (estimated_clean / len(non_empty_lines)) * 100 if non_empty_lines else 0
    
    print(f"   Dòng gốc (không rỗng):     {len(non_empty_lines):,}")
    print(f"   Dự kiến sau làm sạch:      ~{estimated_clean:,} câu")
    print(f"   Tỷ lệ giữ lại:             ~{retention_rate:.1f}%")
    
    if retention_rate < 50:
        print(f"\n   ⚠️  CẢNH BÁO: Có thể mất > 50% dữ liệu!")
        print(f"       Cân nhắc nới lỏng tiêu chí lọc.")
    elif retention_rate < 70:
        print(f"\n   ⚠️  Lưu ý: Sẽ mất ~{100-retention_rate:.0f}% dữ liệu")
    else:
        print(f"\n   ✅ Tốt! Giữ được phần lớn dữ liệu")
    
    # ========== KẾT LUẬN ==========
    print(f"\n{'='*70}")
    print("✅ HOÀN THÀNH PHÂN TÍCH")
    print("="*70)
    print(f"\n📌 Bước tiếp theo: Chạy script 2_clean_data.py để làm sạch\n")
    
    return True


# ========== MAIN ==========
if __name__ == "__main__":
    # Đường dẫn tương đối từ backend/scripts/
    RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "dataset.txt"
    
    print("\n🚀 BẮT ĐẦU PHÂN TÍCH DATASET")
    print(f"📍 Đường dẫn: {RAW_DATA_PATH}")
    
    success = analyze_dataset(RAW_DATA_PATH)
    
    if not success:
        print("\n💡 Hướng dẫn khắc phục:")
        print("   1. Kiểm tra file dataset có tồn tại không:")
        print("      ls backend/data/raw/")
        print("   2. Nếu chưa có, di chuyển file vào đúng chỗ:")
        print("      mv path/to/your/file.txt backend/data/raw/cadao_raw.txt")
        sys.exit(1)