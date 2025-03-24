import pandas as pd
from sqlalchemy import desc
import joblib

import pickle
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import requests
import threading
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app_fastapi = FastAPI(title="API de Prédiction de l'éligibilité au don de sang",
                      description="Cette API permet de prédire l'éligibilité d'une personne au don de sang")

@app_fastapi.get("/")
def great():
    return {"message":" Bienvenue sur l'API pour la prédiction du statut d'elegibilité au don de sang"}
class Donneur_Data(BaseModel):
    
    Genre: str = Field(
        description="Genre du donneur", 
        examples=["Homme", "Femme"],
        default="Homme"
    )
    
    Age: int = Field(
        description="Age du donneur", 
        ge=18, le=65,
        default=30
    )
    
    Niveau_etude: str = Field(
        description="Niveau d'étude du donneur",
        examples=["Primaire", "Secondaire", "Universitaire", "Doctorat"],
        default="Universitaire"
    )
    
    Situation_Matrimoniale: str = Field(
        description="Situation matrimoniale du donneur",
        examples=["Célibataire", "Marié", "Divorcé", "Veuf"],
        default="Célibataire"
    )
    
    Religion: str = Field(
        description="Religion du donneur",
        examples=["Musulman", "Chrétien", "Autre"],
        default="chrétien (catholique)"
    )
    
    A_deja_donne: str = Field(
        description="Le donneur a-t-il déjà donné du sang ?",
        examples=["Oui", "Non"],
        default="Non"
    )
    
    profession: str = Field(
        description="La catégorie socioprofessionnelle de l'individu",
        examples=["Étudiant", "Fonctionnaire", "Indépendant", "Chômeur", "Retraité"],
        default="Étudiant"
    )
    
    Taille: float = Field(
        description="Taille du donneur en cm",
        ge=140, le=210,
        default=175.0
    )
    
    Poids: float = Field(
        description="Poids du donneur en kg",
        ge=45, le=150,
        default=70.0
    )
    
    
    Arrondissement_de_residence: str = Field(
        description="Arrondissement de résidence du donneur",
        examples=["Douala 1", "Douala 2", "Douala 3", "Douala 4", "Douala 5"],
        default="Douala 1"
    )
    
    Quartier_residence: str = Field(
        description="Quartier de résidence du donneur",
        examples=["Bonapriso", "Edea", "Pk16", "Akwa"],
        default="Bonapriso"
    )
    
    
    Nationalite: str = Field(
        description="Nationalité du donneur",
        examples=["Camerounaise", "Sénégalaise", "Ivoirienne", "Gabonaise"],
        default="Camerounaise"
    )
    
    Taux_hemoglobine: float = Field(
        description="Taux d'hémoglobine du donneur (g/dL)",
        ge=8.0, le=20.0,
        default=14.0
    )
    
    class Config:
        schema_extra = {
            "example": {
                "Niveau d'etude": "Universitaire",
                "Genre": "Homme",
                "Taille": 175,
                "Poids": 70,
                "Situation Matrimoniale (SM)": "Célibataire",
                "Profession": "Ingénieur",
                "Arrondissement de résidence": "douala 1",
                "Quartier de Résidence": "Bonapriso",
                "Nationalité": "camerounaise",
                "Religion": "chrétien (catholique)",
                "A-t-il (elle) déjà donné le sang": "Non",
                "Taux d’hémoglobine": 14.0,
                "Age": 30
}
            }
        


class PredictionModel(BaseModel):
    eligible: bool = Field(..., description="Éligibilité au don de sang")
    probability: Optional[float] = Field(None, description="Probabilité d'éligibilité (entre 0 et 1)")
    message: Optional[str] = Field(None, description="Message explicatif")


# Chargement du modèle préentraîné
@app_fastapi.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load('model_random.pkl')
        print("Modèle chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        # Créer un modèle fictif pour le développement
        import sklearn.ensemble
        model = sklearn.ensemble.RandomForestClassifier()
        model.fit([[0, 0, 0, 0]], [1])  # Pour éviter les erreurs pendant le développement
        print("Modèle fictif créé pour le développement")


# Fonction de préparation des données pour la prédiction
def prepare_data_for_prediction(data: Donneur_Data):
    """Prépare les données pour la prédiction en les transformant en DataFrame
    avec le format attendu par le modèle"""
    
    # Conversion en dictionnaire puis en DataFrame
    data_dict = data.dict()
    df = pd.DataFrame([data_dict])
    
    mapping = {
        "Genre": "Genre",
        "Age": "Age",
        "Niveau_etude": "Niveau d'etude",
        "Situation_Matrimoniale": "Situation Matrimoniale (SM)",
        "Religion": "Religion",
        "A_deja_donne": "A-t-il (elle) déjà donné le sang",
        "profession": "Profession_Commune",
        "Taille": "Taille",
        "Poids": "Poids",
        "Quartier_residence": "Quartier de Résidence",
        
        "Arrondissement_de_residence": "Arrondissement de résidence",
        "Nationalite": "Nationalité",
        "Taux_hemoglobine": "Taux d’hémoglobine"
    }

    # Code pour renommer les colonnes du DataFrame
    df = df.rename(columns=mapping)
    
    colonnes_a_conserver = [ 'Niveau d\'etude', 'Genre', 'Taille', 'Poids',
        'Situation Matrimoniale (SM)', 'Profession_Commune',
        'Arrondissement de résidence', 'Quartier de Résidence',
        'Religion', 'A-t-il (elle) déjà donné le sang',
        'Taux d’hémoglobine',
        'ÉLIGIBILITÉ AU DON.', 'Age'
    ]
    
    df = df[colonnes_a_conserver]
    
    # Ici, vous pourriez ajouter des étapes de prétraitement supplémentaires
    # comme la normalisation ou l'encodage des variables catégorielles
    # si cela n'est pas déjà géré par votre pipeline de modèle
    
    return df


@app_fastapi.post("/predict", response_model=PredictionModel)
async def predict(data: Donneur_Data):
    try:
        # Préparation des données
        df = prepare_data_for_prediction(data)
        
        seuil_optimal = 0.65
        
        # Prédiction
        prediction = (modele.predict_proba(X_test)[::,1]>seuil_optimal).astype(int)
        
        # Calcul de la probabilité si disponible
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(df)[:, 1]
        
        # Création du message
        if prediction[0]:
            message = "Éligible au don de sang."
            # Vous pouvez ajouter des informations supplémentaires ici
            if probability is not None and probability[0] > 0.9:
                message += " Excellent candidat pour le don de sang."
            elif probability is not None and probability[0] > 0.7:
                message += " Bon candidat pour le don de sang."
        else:
            message = "Non éligible au don de sang."
            # Vous pouvez ajouter des suggestions ou raisons potentielles ici
            if data.Age < 18:
                message += " L'âge minimum requis est de 18 ans."
            elif data.Age > 65:
                message += " L'âge maximum autorisé est de 65 ans."
            elif data.Taux_hemoglobine < 12:
                message += " Le taux d'hémoglobine est trop bas."
        
        # Préparation de la réponse
        response = PredictionModel(
            eligible=bool(prediction[0]),
            probability=float(probability[0]) if probability is not None else None,
            message=message
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")


