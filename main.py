from src.model import ModelPose





def main(min_detection_confidence=0.5,min_tracking_confidence=0.5):
    model = ModelPose()
    return model.run(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
   


if __name__=="__main__":
    main()
