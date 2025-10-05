# Transformer Decoder Only (from scratch)

Implementasi sederhana arsitektur **Transformer Decoder Only** menggunakan Python dan NumPy, tanpa library deep learning eksternal.  
Komponen yang diimplementasikan:
- Token Embedding  
- Positional Encoding (sinusoidal)  
- Scaled Dot-Product Attention  
- Multi-Head Attention  
- Feed Forward Network  
- Residual Connection + LayerNorm  
- Causal Masking  
- Output Layer (logits + softmax)  

## Dependensi
- Python 3.8+  
- NumPy  

Instalasi NumPy:
```bash
pip install numpy
```

## Cara Menjalankan

Clone repository ini, lalu jalankan program utama:
```bash
git clone https://github.com/username/transformer-from-scratch.git
cd transformer-from-scratch
python main.py
```

## Output Contoh

Program akan menampilkan tensor acak (tokens), bentuk tensor hasil perhitungan (logits, distribusi probabilitas), dan debug dari attention. Contoh:

