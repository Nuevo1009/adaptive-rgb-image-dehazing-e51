from pathlib import Path

import torch

from PIL import Image

from torchvision import transforms

from classifier.model import HazeCNN


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 128


# ============================================================
# DEVICE
# ============================================================

def get_device():

    return torch.device(

        "cuda"

        if torch.cuda.is_available()

        else

        "cpu"

    )


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(

        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )

    ),

    transforms.ToTensor()

])


# ============================================================
# LOAD TRAINED CNN MODEL
# ============================================================

def load_model(

    model_path,

    device=None

):

    model_path = Path(

        model_path

    )


    if not model_path.exists():

        raise FileNotFoundError(

            f"\nModel not found:\n{model_path}"

        )


    if device is None:

        device = get_device()


    print(

        "Loading CNN checkpoint..."

    )


    checkpoint = torch.load(

        model_path,

        map_location=device

    )


    # --------------------------------------------------------
    # CREATE THE SAME CNN ARCHITECTURE
    # --------------------------------------------------------

    model = HazeCNN().to(

        device

    )


    # --------------------------------------------------------
    # CHECK CHECKPOINT FORMAT
    # --------------------------------------------------------

    if isinstance(

        checkpoint,

        dict

    ) and (

        "model_state_dict"

        in checkpoint

    ):


        print(

            "Checkpoint format detected: model_state_dict"

        )


        state_dict = checkpoint[

            "model_state_dict"

        ]


    else:


        print(

            "Checkpoint format detected: direct state_dict"

        )


        state_dict = checkpoint


    # --------------------------------------------------------
    # LOAD WEIGHTS
    # --------------------------------------------------------

    model.load_state_dict(

        state_dict,

        strict=True

    )


    model.eval()


    print(

        "CNN model loaded successfully."

    )


    return model, device


# ============================================================
# CNN HAZE SCORE PREDICTION
# ============================================================

def predict_haze_score(

    image_path,

    model,

    device

):


    image_path = Path(

        image_path

    )


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(

        image_path

    ).convert(

        "RGB"

    )


    # --------------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------------

    image_tensor = transform(

        image

    )


    image_tensor = image_tensor.unsqueeze(

        0

    ).to(

        device

    )


    # --------------------------------------------------------
    # CNN PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():


        score = model(

            image_tensor

        ).item()


    return float(

        score

    )


# ============================================================
# FUZZY MEMBERSHIP FUNCTIONS
# ============================================================

def triangular_membership(

    x,

    a,

    b,

    c

):

    """
    Triangular fuzzy membership function.

          /\
         /  \
        /    \

    a    b    c
    """

    if x <= a:

        return 0.0


    elif a < x <= b:


        return (

            x - a

        ) / (

            b - a

        )


    elif b < x < c:


        return (

            c - x

        ) / (

            c - b

        )


    else:

        return 0.0


# ============================================================
# FUZZY MEMBERSHIP CALCULATION
# ============================================================

def calculate_fuzzy_memberships(

    haze_score

):


    # --------------------------------------------------------
    # CLEAR MEMBERSHIP
    #
    # High when haze score is close to 0
    # --------------------------------------------------------

    if haze_score <= 0.3:

        clear_membership = 1.0


    elif haze_score < 0.5:


        clear_membership = (

            0.5 - haze_score

        ) / 0.2


    else:

        clear_membership = 0.0


    # --------------------------------------------------------
    # UNCERTAIN MEMBERSHIP
    #
    # Peak around haze score = 0.5
    # --------------------------------------------------------

    uncertain_membership = triangular_membership(

        haze_score,

        0.2,

        0.5,

        0.8

    )


    # --------------------------------------------------------
    # HAZY MEMBERSHIP
    #
    # High when haze score is close to 1
    # --------------------------------------------------------

    if haze_score <= 0.5:

        hazy_membership = 0.0


    elif haze_score < 0.7:


        hazy_membership = (

            haze_score - 0.5

        ) / 0.2


    else:

        hazy_membership = 1.0


    return {

        "clear":

            float(
                clear_membership
            ),

        "uncertain":

            float(
                uncertain_membership
            ),

        "hazy":

            float(
                hazy_membership
            )

    }


# ============================================================
# FUZZY DECISION
# ============================================================

def fuzzy_classification(

    haze_score

):


    memberships = (

        calculate_fuzzy_memberships(

            haze_score

        )

    )


    # --------------------------------------------------------
    # SELECT CLASS WITH HIGHEST MEMBERSHIP
    # --------------------------------------------------------

    classification = max(

        memberships,

        key=memberships.get

    )


    return (

        classification.upper(),

        memberships

    )


# ============================================================
# FINAL IMAGE CLASSIFICATION
# ============================================================

def classify_image(

    image_path,

    model,

    device

):


    # --------------------------------------------------------
    # STEP 1:
    # GET CNN HAZE SCORE
    # --------------------------------------------------------

    haze_score = (

        predict_haze_score(

            image_path,

            model,

            device

        )

    )


    # --------------------------------------------------------
    # STEP 2:
    # FUZZY CLASSIFICATION
    # --------------------------------------------------------

    classification, memberships = (

        fuzzy_classification(

            haze_score

        )

    )


    return (

        classification,

        haze_score,

        memberships

    )