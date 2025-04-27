import pandas as pd
from anomaly_detector import detect_anomalies
from rag_generator import explain_anomaly
from embedder import get_embedding
from faiss_index import build_faiss_index

def main():
    # Load structured data
    data = pd.read_csv("data/structured_data.csv")

    # Detect anomalies
    anomalies = detect_anomalies(data)
    print(f"Detected {len(anomalies)} anomalies.")

    # Build FAISS index from guidelines (one-time step)
    guidelines_text = open("data/docs/anomaly_guidelines.txt").read()
    documents = [{"text": guidelines_text, "embedding": get_embedding(guidelines_text)}]
    build_faiss_index(documents)

    # Explain each anomaly
    for idx, anomaly in anomalies.iterrows():
        anomaly_desc = anomaly.to_json()
        explanation = explain_anomaly(anomaly_desc)
        print(f"\nAnomaly at index {idx}:")
        print(explanation)

if __name__ == "__main__":
    main()
