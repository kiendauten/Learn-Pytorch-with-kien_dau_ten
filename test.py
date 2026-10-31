import os
import sys
from collections import Counter
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Tải mô hình spaCy và SentenceTransformer
# Nếu máy chưa tải: 
# pip install spacy sentence-transformers scikit-learn
# python -m spacy download en_core_web_sm
print("Đang khởi tạo mô hình NLP...")
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_meaningful_lemmas(text: str):
    """Tách từ, đưa về dạng nguyên mẫu (lemma), lọc stop words và chỉ giữ lại thực từ."""
    doc = nlp(text.lower())
    lemmas = []
    for token in doc:
        if token.is_punct or token.is_space or token.is_stop or len(token.text) <= 2:
            continue
        # Giữ lại Danh từ, Động từ, Tính từ mang ý nghĩa nội dung
        if token.pos_ in ["NOUN", "VERB", "ADJ"]:
            lemmas.append(token.lemma_)
    return lemmas


def cluster_synonyms(unique_words, word_freq, similarity_threshold=0.62):
    """Gom các từ có độ tương đồng ngữ nghĩa cosin >= similarity_threshold vào cùng một nhóm."""
    if not unique_words:
        return []

    embeddings = model.encode(unique_words, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)

    visited = set()
    clusters = []

    # Ưu tiên từ xuất hiện nhiều nhất làm tâm cụm
    sorted_words = sorted(unique_words, key=lambda w: word_freq[w], reverse=True)
    word_to_idx = {w: i for i, w in enumerate(unique_words)}

    for word in sorted_words:
        if word in visited:
            continue

        idx = word_to_idx[word]
        current_cluster = [word]
        visited.add(word)

        for other_word in sorted_words:
            if other_word not in visited:
                other_idx = word_to_idx[other_word]
                if sim_matrix[idx][other_idx] >= similarity_threshold:
                    current_cluster.append(other_word)
                    visited.add(other_word)

        clusters.append(current_cluster)

    return clusters


def process_file(
    input_path="1.inp", output_path="1.out", similarity_threshold=0.62
):
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file {input_path}")
        return

    # Đọc file 1.inp: dòng đầu là k, phần còn lại là text
    with open(input_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line:
            print("File rỗng!")
            return
        try:
            k = int(first_line.strip())
        except ValueError:
            print("Dòng đầu tiên của 1.inp phải là số nguyên k.")
            return

        text = f.read()

    # Xử lý NLP
    lemmas = extract_meaningful_lemmas(text)
    word_freq = Counter(lemmas)
    unique_words = list(word_freq.keys())

    clusters = cluster_synonyms(
        unique_words, word_freq, similarity_threshold=similarity_threshold
    )

    # Lọc nhóm có tổng số lần xuất hiện > k
    results = []
    for cluster in clusters:
        total_count = sum(word_freq[w] for w in cluster)
        if total_count > k:
            results.append((cluster, total_count))

    # Sắp xếp giảm dần theo tổng tần suất
    results.sort(key=lambda x: x[1], reverse=True)

    # Ghi ra file 1.out
    with open(output_path, "w", encoding="utf-8") as out:
        if not results:
            out.write(f"Khong co nhom tu nao xuat hien > {k} lan.\n")
        else:
            for cluster, total_count in results:
                # Format: [từ đại diện] (tổng count lần) -> từ1 (count1), từ2 (count2)
                words_detail = ", ".join(
                    [f"{w} ({word_freq[w]})" for w in cluster]
                )
                out.write(
                    f"[{cluster[0]}] - Tong {total_count} lan: {words_detail}\n"
                )

    print(
        f"Hoàn thành! Đã xử lý {len(results)} nhóm từ và lưu vào {output_path}"
    )


if __name__ == "__main__":
    process_file("1.inp", "1.out", similarity_threshold=0.62)