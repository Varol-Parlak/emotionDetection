import cv2 
from deepface import DeepFace

cap = cv2.VideoCapture(0)
a = 5
b = 0
last_data = None
threshold = 0.5
while True:
    ret, frame = cap.read()
    if not ret :
        break
    
    if b % a == 0:
        face = DeepFace.analyze(frame,actions=["emotion"],detector_backend="mtcnn", enforce_detection=False,silent=True)
        #detector_backend = opencv(hızlı, kesinlik az), mtcnn(orta hız, orta kesinlik) veya retinaface(çok yavaş, fazla kesinlik) 
        if face and 'dominant_emotion' in face[0]:
            dom_emo = face[0]["dominant_emotion"]
            region = face[0]["region"]
            conf = face[0]["emotion"]     
            score = conf[dom_emo] / 100

            if score >= threshold:
                last_data = {
                    "w": region["w"],
                    "h": region["h"],
                    "x": region["x"],
                    "y": region["y"],
                    "dom_emo": dom_emo,
                    "score_text": f"{score:.2f}"
                }
            else:
                last_data = {
                    "w": region["w"],
                    "h": region["h"],
                    "x": region["x"],
                    "y": region["y"],
                    "dom_emo": "Undecided",
                    "score_text": f"{score:.2f}"
                }

        else:
            last_data = None
    
    if last_data:
        w , h, x, y, dom_emo, score_text = last_data["w"], last_data["h"], last_data["x"], last_data["y"], last_data["dom_emo"], last_data["score_text"]
        
        colour = (0,0,255) if dom_emo == "Undecided" else (0,255,0) 
        cv2.rectangle(frame, (x,y), (x + w, y + h), colour, 2)
        cv2.putText(frame, score_text,(x + w - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX,0.5,colour,1,cv2.LINE_AA)
        cv2.putText(frame, dom_emo, (x + 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2, cv2.LINE_AA)

    cv2.imshow("Deepface", frame)
    b+=1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()