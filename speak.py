import gtts
import playsound

def speakTrash(trash: str):
    if trash == 'cardboard':
        script = "이 쓰레기는 카드보드입니다."
        filename = "cardboard.mp3"
    elif trash == 'glass':
        script = "이 쓰레기는 유리입니다."
        filename = "glass.mp3"
    elif trash == 'metal':
        script = "이 쓰레기는 금속입니다."
        filename = "metal.mp3"
    elif trash == 'paper':
        script = "이 쓰레기는 종이입니다."
        filename = "paper.mp3"
    elif trash == 'plastic':
        script = "이 쓰레기는 플라스틱입니다."
        filename = "plastic.mp3"
    else:
        print("알 수 없는 쓰레기 종류입니다.")
        return

    tts = gtts.gTTS(text=script, lang='ko')
    tts.save(filename)
    playsound.playsound(filename)

# 예시 호출
speakTrash('plastic')