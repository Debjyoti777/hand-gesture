from asl_module.asl_recognize_pc import ASLLetterRecognizer

recognizer = ASLLetterRecognizer(
    model_dir="asl_module/asl_data"
)

recognizer.run_recognition()