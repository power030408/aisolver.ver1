# AISolver

수동 스크린샷 없이 화면을 직접 캡처해 문제를 자동 인식하고, OCR과 AI로 단계별 풀이를 제공하는 데스크톱 문제 해결 도우미입니다.

## Screenshots

### Main Dashboard

![Main dashboard](assets/%EC%82%AC%EC%9A%A9%EC%9E%90%20%EC%B2%A8%EB%B6%80%20%ED%8C%8C%EC%9D%BC1.png)

### Region Selection

![Region selection](assets/%EC%82%AC%EC%9A%A9%EC%9E%90%20%EC%B2%A8%EB%B6%80%20%ED%8C%8C%EC%9D%BC3.png)

### AI Result

![AI result](assets/%EC%82%AC%EC%9A%A9%EC%9E%90%20%EC%B2%A8%EB%B6%80%20%ED%8C%8C%EC%9D%BC.png)

## Features

- Direct screen capture without saving screenshots manually
- OCR-based question text extraction
- AI-generated step-by-step explanations
- Image and PDF question input support
- Tesseract OCR auto-detection with optional bundled OCR files
- Windows/macOS PyInstaller build scripts

## Requirements

- Python 3.10+
- OpenAI API key or Google Gemini API key
- Optional: Tesseract OCR for local OCR

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python ai_solver.py
```

## Build

Windows:

```powershell
.\build_ai_solver.ps1
```

macOS:

```bash
./build_ai_solver_mac.sh
```
