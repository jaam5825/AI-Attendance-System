from deepface import DeepFace

for i in range(9):  # 0↔1 through 8↔9, so you need 0.jpg to 9.jpg
    img1 = f"dataset/imran/{i}.jpg"
    img2 = f"dataset/imran/{i+1}.jpg"

    try:
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name="Facenet",        # match your attendance system
            enforce_detection=False
        )

        verified = result["verified"]
        distance = round(result["distance"], 4)
        threshold = result.get("threshold", "N/A")
        print(f"{i}.jpg vs {i+1}.jpg -> Verified: {verified} | Distance: {distance} | Threshold: {threshold}")

    except Exception as e:
        print(f"Error comparing {i}.jpg and {i+1}.jpg: {e}")