import face_recognition
#臉部驗證程式

#比對joyce
# known_image = face_recognition.load_image_file("./rec/joyce.jpg")
# unknown_image = face_recognition.load_image_file("who2.jpg")
#比對安海瑟威
#known_image = face_recognition.load_image_file("./rec/Anne-Hathaway.jpg")
#unknown_image = face_recognition.load_image_file("who.jpg")
#比對艾瑪華生
known_image = face_recognition.load_image_file("./rec/Emma-watson.jpg")
unknown_image = face_recognition.load_image_file("who3.jpg")
#艾瑪華生與馬英九
# known_image = face_recognition.load_image_file("./rec/Emma-watson.jpg")
# unknown_image = face_recognition.load_image_file("ma.jpg")


biden_encoding = face_recognition.face_encodings(known_image)[0]

unknown_image = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding],unknown_image)

print(results)

