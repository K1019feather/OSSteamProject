import gtts
import playsound
import os
import time

def speakTrash(trash: str):
    script_map = {
        'cardboard': ("이 쓰레기는 판지입니다.", "cardboard.mp3"),
        'glass': ("이 쓰레기는 유리입니다.", "glass.mp3"),
        'metal': ("이 쓰레기는 금속입니다.", "metal.mp3"),
        'paper': ("이 쓰레기는 종이입니다.", "paper.mp3"),
        'plastic': ("이 쓰레기는 플라스틱입니다.", "plastic.mp3"),
    }

    if trash not in script_map:
        print("알 수 없는 쓰레기 종류입니다.")
        return

    script, filename = script_map[trash]

    # gTTS로 mp3 생성
    tts = gtts.gTTS(text=script, lang='ko')
    tts.save(filename)

    # 약간 대기 (파일 저장 완료 보장용)
    time.sleep(0.5)

    try:
        playsound.playsound(filename)
    except Exception as e:
        print(f"재생 오류: {e}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)  # 파일 자동 삭제
