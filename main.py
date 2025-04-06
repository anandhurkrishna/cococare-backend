from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ✅ Constants
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# ✅ In-memory user storage (for testing)
users_db = {}

# ✅ Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load the trained model
MODEL_PATH = "cococare_model.keras"  # Update this path
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Define class labels
CLASS_LABELS = [
    "CCI_Caterpillars", "CCI_Leaflets", "Healthy_Leaves",
    "WCLWD_DryingofLeaflets", "WCLWD_Flaccidity", "WCLWD_Yellowing"
]

# ✅ Mock database to store predictions
predictions_db = []


# ✅ Hash password function
def hash_password(password: str):
    return pwd_context.hash(password)


# ✅ Authenticate user from token
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in users_db:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# ✅ User Registration
@app.post("/register")
async def register(username: str, password: str):
    if username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed_password = hash_password(password)
    users_db[username] = {"username": username, "hashed_password": hashed_password}
    
    return {"message": "User registered successfully"}


# ✅ User Login (Token Generation)
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    access_token = jwt.encode(
        {"sub": form_data.username, "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)},
        SECRET_KEY, algorithm=ALGORITHM
    )

    return {"access_token": access_token, "token_type": "bearer"}


# ✅ Image preprocessing function
def preprocess_image(image: Image.Image):
    """Preprocess the image for model prediction"""
    image = image.resize((224, 224))  # Resize to match the model's input size
    img_array = np.array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# ✅ Prediction Route (Protected)
@app.post("/predict")
async def predict(file: UploadFile = File(...), username: str = Depends(get_current_user)):
    try:
        # Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess the image
        img_array = preprocess_image(image)

        # Make a prediction
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions))

        # Get the predicted label
        predicted_label = CLASS_LABELS[np.argmax(predictions)]

        # Save the prediction in the mock database
        prediction_result = {
            "username": username,
            "image_name": file.filename,
            "disease": predicted_label,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        predictions_db.append(prediction_result)

        return {
            "image_name": file.filename,
            "disease": predicted_label,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


# ✅ Results Endpoint (Protected)
@app.get("/results")
async def get_results(username: str = Depends(get_current_user)):
    """Returns saved predictions for the logged-in user"""
    user_predictions = [pred for pred in predictions_db if pred["username"] == username]

    if not user_predictions:
        return {"message": "No predictions saved yet."}

    return user_predictions


# ✅ Main Entry Point
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1",port=8000)