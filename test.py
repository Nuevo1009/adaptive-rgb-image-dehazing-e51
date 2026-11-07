from pathlib import Path

from classifier.predict import (

    load_model,

    classify_image

)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(

    __file__

).resolve().parent


MODEL_PATH = (

    BASE_DIR
    /
    "models"
    /
    "best_haze_CNN.pth"

)


TEST_FOLDER = (

    BASE_DIR
    /
    "dataset"
    /
    "CNN_Dataset"
    /
    "test"

)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

VALID_EXTENSIONS = (

    ".jpg",

    ".jpeg",

    ".png",

    ".bmp",

    ".tif",

    ".tiff"

)


# ============================================================
# MAIN
# ============================================================

def main():


    print(

        "\n============================================"

    )


    print(

        " CNN IMAGE TESTING"

    )


    print(

        "============================================"

    )


    model, device = load_model(

        MODEL_PATH

    )


    image_files = []


    for class_folder in [

        "clear",

        "hazy"

    ]:


        folder = (

            TEST_FOLDER
            /
            class_folder

        )


        if folder.exists():


            image_files.extend([

                file

                for file in folder.iterdir()

                if file.suffix.lower()

                in VALID_EXTENSIONS

            ])


    print(

        f"\nImages found: "

        f"{len(image_files)}"

    )


    for image_path in image_files:


        classification, score, memberships = classify_image(

            image_path,

            model,

            device

        )
        print(f"\nImage: {image_path.name}")

        print(f"CNN Haze Score: {score:.4f}")

        print(f"Clear Membership: {memberships['clear']:.4f}")

        print(f"Uncertain Membership: {memberships['uncertain']:.4f}")

        print(f"Hazy Membership: {memberships['hazy']:.4f}")

        print(f"Final Classification: {classification}")
    

if __name__ == "__main__":

    main()