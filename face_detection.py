# Import library yang diperlukan
import cv2
from deepface import DeepFace
import sys
import os

print("Memulai program deteksi ekspresi wajah...")

# Inisialisasi kamera (0 adalah ID default untuk webcam bawaan)
# Jika Anda punya lebih dari satu kamera, Anda bisa coba 1, 2, dst.
cap = cv2.VideoCapture(0)

# Periksa apakah kamera berhasil dibuka
if not cap.isOpened():
    print("Error: Tidak dapat membuka webcam.")
    print("Pastikan kamera tidak sedang digunakan oleh aplikasi lain dan driver sudah terinstal.")
    print("Jika menggunakan laptop, pastikan izin kamera diberikan ke aplikasi terminal/IDE Anda.")
    sys.exit() # Keluar dari program jika kamera tidak bisa dibuka

print("Webcam berhasil dibuka. Mendeteksi ekspresi wajah secara real-time...")
print("Tekan tombol 'q' atau 'ESC' untuk keluar dari aplikasi.")

# --- Fungsi untuk menggambar teks dan kotak dengan border ---
def draw_text_with_background(img, text, x, y, font_scale, thickness, text_color, bg_color):
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    # Background rectangle
    cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)
    # Text
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

# --- Loop Utama untuk Deteksi Real-time ---
print("Proses deteksi akan dimulai. Model DeepFace akan diunduh saat pertama kali dibutuhkan (ini bisa memakan waktu tergantung koneksi internet Anda).")
print("Harap bersabar jika ada jeda di awal.")

while True:
    # Baca satu frame dari kamera
    ret, frame = cap.read()

    # Jika frame tidak berhasil dibaca (misalnya kamera terputus)
    if not ret:
        print("Gagal mengambil frame dari kamera. Mengakhiri program.")
        break

    # DeepFace akan melakukan analisis ekspresi wajah
    try:
        # analyze() mengembalikan list dari dict, satu dict per wajah yang terdeteksi
        # Saat pertama kali baris ini dieksekusi, DeepFace akan mengunduh model
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Iterasi untuk setiap wajah yang terdeteksi
        for result in results:
            # Ambil emosi dominan dan koordinat wajah
            dominant_emotion = result['dominant_emotion']
            region = result['region']

            # Gambar bounding box di sekitar wajah
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Tentukan warna bounding box berdasarkan emosi dominan (opsional, untuk estetika)
            color = (0, 255, 0) # Default Hijau
            if dominant_emotion == 'sad':
                color = (255, 0, 0) # Biru
            elif dominant_emotion == 'angry':
                color = (0, 0, 255) # Merah
            elif dominant_emotion == 'happy':
                color = (0, 255, 255) # Kuning (Cinta)
            elif dominant_emotion == 'surprise':
                color = (255, 255, 0) # Cyan (Terkejut)
            elif dominant_emotion == 'fear':
                color = (128, 0, 128) # Ungu (Takut)
            elif dominant_emotion == 'disgust':
                color = (0, 128, 128) # Teal (Jijik)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) # Gambar kotak

            # Tulis ekspresi yang terdeteksi di atas bounding box
            text_emotion = f"Emosi: {dominant_emotion}"
            draw_text_with_background(frame, text_emotion, x, y - 10, 0.7, 2, color, (0,0,0)) # Teks emosi

            # Opsional: Tulis probabilitas setiap emosi
            # Dapatkan probabilitas dalam persentase, urutkan dari yang tertinggi
            emotions_probs = sorted(result['emotion'].items(), key=lambda item: item[1], reverse=True)
            y_offset = y + h + 20 # Posisi awal untuk daftar probabilitas
            
            # Tampilkan 3 emosi teratas (termasuk yang dominan jika Anda mau, atau hanya yang lain)
            # Contoh di bawah ini menampilkan 3 emosi dengan probabilitas tertinggi
            for i, (emotion_name, prob) in enumerate(emotions_probs):
                if i < 3: # Tampilkan 3 emosi dengan probabilitas tertinggi
                    text_prob = f"{emotion_name}: {prob:.1f}%" # Format 1 angka desimal
                    draw_text_with_background(frame, text_prob, x, y_offset, 0.5, 1, (255, 255, 255), (50,50,50)) # Putih dengan latar abu-abu
                    y_offset += 20

    except Exception as e:
        # Jika ada error saat analisis (misalnya tidak ada wajah terdeteksi,
        # atau DeepFace sedang mengunduh model pertama kali dan belum siap)
        # Kita bisa menampilkan pesan di konsol atau mengabaikannya agar tidak spam
        # print(f"Analisis DeepFace bermasalah: {e}")
        pass # Abaikan error agar program tetap berjalan tanpa crash

    # Tampilkan frame di jendela
    cv2.imshow('Deteksi Ekspresi Wajah - Tekan Q atau ESC untuk Keluar', frame)

    # Tunggu tombol 'q' (ASCII 113) atau 'ESC' (ASCII 27) ditekan untuk keluar dari loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27: # 27 adalah kode ASCII untuk ESC
        break

# Setelah loop berakhir, bebaskan resource kamera dan tutup semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()
print("Program deteksi ekspresi wajah selesai.")